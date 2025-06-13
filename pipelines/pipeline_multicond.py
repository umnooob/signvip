# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, deprecate, is_accelerate_available, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomGrayscale,
    RandomResizedCrop,
    ToTensor,
)
from tqdm import tqdm
from transformers import CLIPImageProcessor

from models.mutual_self_attention import ReferenceAttentionControl

from .context import get_context_scheduler

logger = logging.get_logger(__name__)


@dataclass
class SignViPPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class SignViPPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        appearance_encoder,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        condition_encoder=None,
        image_proj_model=None,
        empty_text_emb=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            appearance_encoder=appearance_encoder,
            denoising_unet=denoising_unet,
            condition_encoder=condition_encoder,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
        )
        self.empty_text_emb = empty_text_emb
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.image_transform = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.denoising_unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(
            self.denoising_unet, "_hf_hook"
        ):
            return self.device
        for module in self.denoising_unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(
            range(latents.shape[0]), desc="Decoding latents into frames"
        ):
            image = self.vae.decode(
                latents[frame_idx : frame_idx + 1].to(self.vae.device)
            ).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().float()
            video.append(image)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)

        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        # If empty_text_emb is provided and text_encoder is not, use only empty_text_emb
        if self.empty_text_emb is not None and not hasattr(self, "text_encoder"):
            text_embeddings = self.empty_text_emb.to(device)
            if len(text_embeddings.shape) == 2:  # Add batch dimension if needed
                text_embeddings = text_embeddings.unsqueeze(0)
            batch_size = 1
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )
            if do_classifier_free_guidance:
                text_embeddings = torch.cat([text_embeddings, text_embeddings])
            return text_embeddings

        # Original implementation for when text_encoder and tokenizer are provided
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )
        if do_classifier_free_guidance:
            if self.empty_text_emb is not None:
                uncond_embeddings = self.empty_text_emb.to(device)
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(
                    batch_size * num_videos_per_prompt, 1, 1
                )
            else:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:{prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt
                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None
                uncond_embeddings = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                uncond_embeddings = uncond_embeddings[0]
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = uncond_embeddings.repeat(
                    1, num_videos_per_prompt, 1
                )
                uncond_embeddings = uncond_embeddings.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    # def interpolate_latents(
    #     self, latents: torch.Tensor, interpolation_factor: int, device
    # ):
    #     if interpolation_factor < 2:
    #         return latents

    #     new_latents = torch.zeros(
    #         (
    #             latents.shape[0],
    #             latents.shape[1],
    #             ((latents.shape[2] - 1) * interpolation_factor) + 1,
    #             latents.shape[3],
    #             latents.shape[4],
    #         ),
    #         device=latents.device,
    #         dtype=latents.dtype,
    #     )

    #     org_video_length = latents.shape[2]
    #     rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

    #     new_index = 0

    #     v0 = None
    #     v1 = None

    #     for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
    #         v0 = latents[:, :, i0, :, :]
    #         v1 = latents[:, :, i1, :, :]

    #         new_latents[:, :, new_index, :, :] = v0
    #         new_index += 1

    #         for f in rate:
    #             v = get_tensor_interpolation_method()(
    #                 v0.to(device=device), v1.to(device=device), f
    #             )
    #             new_latents[:, :, new_index, :, :] = v.to(latents.device)
    #             new_index += 1

    #     new_latents[:, :, new_index, :, :] = v1
    #     new_index += 1

    #     return new_latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image,
        sk_images,
        hamer_images,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        video_length=None,
        num_images_per_prompt=1,
        eta: float = 0.0,
        condition_encoder=None,
        pose_latent=None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=12,
        context_stride=1,
        context_overlap=2,
        context_batch_size=1,
        interpolation_factor=1,
        reference_control_writer=None,
        reference_control_reader=None,
        empty_text_emb=None,
        **kwargs,
    ):
        # Allow passing empty_text_emb at call time (overrides self.empty_text_emb)
        if empty_text_emb is not None:
            self.empty_text_emb = empty_text_emb
        condition_encoder = condition_encoder or self.condition_encoder
        if sk_images is not None:
            video_length = video_length or sk_images.shape[2]
        elif pose_latent is not None:
            video_length = pose_latent.shape[0]
        else:
            raise ValueError("Either sk_images or pose_latent must be provided.")
        # Default height and width to unet
        height = (
            height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        )
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        if reference_control_writer is None:
            reference_control_writer = ReferenceAttentionControl(
                self.appearance_encoder,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="write",
                batch_size=context_batch_size,
                fusion_blocks="full",
            )

        if reference_control_reader is None:
            reference_control_reader = ReferenceAttentionControl(
                self.denoising_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="read",
                batch_size=context_batch_size,
                fusion_blocks="full",
            )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            condition_encoder.dtype,
            device,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = (
            ref_image_latents * self.vae.config.scaling_factor
        )  # (b, 4, h, w)

        text_embeddings = self._encode_prompt(
            "",
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            "",
        )
        text_embeddings = text_embeddings.repeat(context_batch_size, 1, 1)

        if pose_latent is not None:
            pose_latent = pose_latent.to(
                dtype=condition_encoder.dtype, device=self.device
            )
        else:
            if isinstance(sk_images, torch.Tensor):
                sk_cond_tensor = sk_images.to(
                    dtype=condition_encoder.dtype, device=self.device
                )  # (bs, c, t, h, w)
            else:
                sk_cond_tensor_list = []
                for sk_image in sk_images:
                    sk_image = self.ref_image_processor.preprocess(
                        sk_image, height=height, width=width
                    )  # [b, c, h, w]
                    sk_image = sk_image.unsqueeze(2)  # (bs, c, 1, h, w)
                    sk_cond_tensor_list.append(sk_image)
                sk_cond_tensor = torch.cat(sk_cond_tensor_list, dim=2).to(
                    dtype=condition_encoder.dtype, device=self.device
                )  # (bs, c, t, h, w)

            if isinstance(sk_images, torch.Tensor):
                hamer_cond_tensor = hamer_images.to(
                    dtype=condition_encoder.dtype, device=self.device
                )  # (bs, c, t, h, w)
            else:
                hamer_cond_tensor_list = []
                for hamer_image in hamer_images:
                    hamer_image = self.ref_image_processor.preprocess(
                        hamer_image, height=height, width=width
                    )  # [b, c, h, w]
                    hamer_image = hamer_image.unsqueeze(2)  # (bs, c, 1, h, w)
                    hamer_cond_tensor_list.append(hamer_image)
                hamer_cond_tensor = torch.cat(hamer_cond_tensor_list, dim=2).to(
                    dtype=condition_encoder.dtype, device=self.device
                )  # (bs, c, t, h, w)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (
                        latents.shape[0] * (2 if do_classifier_free_guidance else 1),
                        *latents.shape[1:],
                    ),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # 1. Forward reference image
                self.appearance_encoder(
                    ref_image_latents.repeat(
                        (2 if do_classifier_free_guidance else 1) * context_batch_size,
                        1,
                        1,
                        1,
                    ),
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )
                reference_control_reader.update(
                    reference_control_writer, dtype=self.denoising_unet.dtype
                )

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    batch_contexts = context_queue[
                        i * context_batch_size : (i + 1) * context_batch_size
                    ]
                    # Pad the last batch if necessary
                    if len(batch_contexts) < context_batch_size:
                        batch_contexts.extend(
                            [batch_contexts[-1]]
                            * (context_batch_size - len(batch_contexts))
                        )
                    global_context.append(batch_contexts)

                for context in global_context:
                    # 3.1 expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        torch.cat([latents[:, :, c] for c in context])
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )
                    b, c, f, h, w = latent_model_input.shape
                    if pose_latent is not None:
                        pose_latents_input = torch.cat(
                            [pose_latent[c] for c in context]
                        ).to(device)
                        sk_features = condition_encoder.infer(
                            pose_latents_input, video_length=f
                        )
                    else:

                        sk_cond_input = torch.cat(
                            [sk_cond_tensor[:, :, c] for c in context]
                        ).to(device)
                        hamer_cond_input = torch.cat(
                            [hamer_cond_tensor[:, :, c] for c in context]
                        ).to(device)

                        sk_cond_input = rearrange(
                            sk_cond_input, "b c f h w -> (b f) c h w"
                        )
                        hamer_cond_input = rearrange(
                            hamer_cond_input, "b c f h w -> (b f) c h w"
                        )
                        sk_features = condition_encoder(
                            sk_cond_input, hamer_cond_input, video_length=f
                        )
                    sk_features = [
                        rearrange(feature, "(b f) c h w -> b c f h w", b=batch_size)
                        for feature in sk_features
                    ]
                    sk_features = [
                        feature.repeat(
                            2 if do_classifier_free_guidance else 1, 1, 1, 1, 1
                        ).to(self.denoising_unet.dtype)
                        for feature in sk_features
                    ]
                    pred = self.denoising_unet(
                        latent_model_input.to(self.denoising_unet.dtype),
                        t.to(self.denoising_unet.dtype),
                        encoder_hidden_states=text_embeddings.to(
                            self.denoising_unet.dtype
                        ),
                        sk_features=sk_features,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):
                        noise_pred[:, :, c] = noise_pred[:, :, c] + pred
                        counter[:, :, c] = counter[:, :, c] + 1

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                reference_control_reader.clear()
                reference_control_writer.clear()

        reference_control_reader.restore_hacked_forward()
        reference_control_writer.restore_hacked_forward()

        # if interpolation_factor > 0:
        #     latents = self.interpolate_latents(latents, interpolation_factor, device)

        video_tensor = self.decode_latents(latents)
        return video_tensor
