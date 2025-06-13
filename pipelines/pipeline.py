# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from typing import Callable, List, Optional, Union

import torch
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
from diffusers.utils import is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from modules import UNetMotionFramesPEModel
from modules.apperance_encoder import AppearanceEncoderModel
from modules.reference_control import ReferenceAttentionControl
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

from .context import get_context_scheduler

# from utils import get_absolute_sinusoidal_encoding


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class T2SLVPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        denoising_unet: UNetMotionFramesPEModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        tokenizer=None,
        text_encoder=None,
        appearance_encoder: AppearanceEncoderModel = None,
        max_sin_pe_len=1000,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            denoising_unet=denoising_unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            appearance_encoder=appearance_encoder,
        )
        self.reference_control_writer = ReferenceAttentionControl(
            self.appearance_encoder, do_classifier_free_guidance=True, mode="write"
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet, do_classifier_free_guidance=True, mode="read"
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.max_sin_pe_len = max_sin_pe_len

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

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
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
            assert (
                latents.shape == shape
            ), f"Latents shape {latents.shape} does not match expected shape {shape}"
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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
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

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_reference_embeds(self, reference_image):
        reference_image_tensor = self.clip_image_processor.preprocess(
            reference_image, return_tensors="pt"
        ).pixel_values
        reference_image_tensor = reference_image_tensor.to(
            dtype=self.dtype, device=self.device
        )
        reference_image_embeds = self.image_encoder(reference_image_tensor).image_embeds
        return reference_image_embeds

    def prepare_reference_latents(self, reference_image):
        reference_image = self.image_processor.preprocess(reference_image).to(
            dtype=self.dtype, device=self.device
        )
        reference_image_latents = self.vae.encode(reference_image).latent_dist.sample()
        reference_image_latents = (
            self.vae.config.scaling_factor * reference_image_latents
        )
        return reference_image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        reference_image,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        strength=1.0,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_overlap=4,
        do_multi_devices_inference=False,
        save_gpu_memory=False,
        negative_prompt=None,
        noise_prior=True,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        batch_size = 1

        # Prepare timesteps
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )

        num_channels_latents = self.denoising_unet.in_channels
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # prepare latents

        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        context_scheduler = get_context_scheduler(context_schedule)
        context_queue = list(
            context_scheduler(
                step=0,
                num_frames=video_length,
                context_size=context_frames,
                context_stride=1,
                context_overlap=context_overlap,
                closed_loop=False,
            )
        )

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.dtype,
            torch.device("cpu"),
            generator,
        )

        ref_image_latents = self.prepare_reference_latents(reference_image)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        frame_encodings = get_absolute_sinusoidal_encoding(
            torch.arange(video_length).unsqueeze(0).repeat(batch_size, 1),
            video_length,
            dim=self.denoising_unet.config.time_cond_proj_dim,
        ).to(device, self.dtype)
        with self.progress_bar(total=len(context_queue)) as context_progress_bar:
            for context_idx, context in enumerate(context_queue):
                # latent_model_input = [context images (not in first context), noisy images]
                if context_idx == 0:
                    mask = 0
                else:
                    mask = context_overlap

                input_latents = latents[:, :, context, ...].to(
                    device
                )  # [bs, c, f, w, h]

                with self.progress_bar(total=len(timesteps)) as timestep_progress_bar:
                    for i, t in enumerate(timesteps):

                        latent_model_input = input_latents  # [bs, c, f, w, h]
                        latent_model_input = latent_model_input.repeat(
                            2 if do_classifier_free_guidance else 1, 1, 1, 1, 1
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        # frame encoding
                        context_frame_encodings = frame_encodings[
                            :, context, ...
                        ]  # [bs,f,dim]

                        context_frame_encodings = context_frame_encodings.repeat(
                            2 if do_classifier_free_guidance else 1, 1, 1
                        )  # [2*bs,f,dim]
                        self.appearance_encoder(
                            ref_image_latents.repeat(
                                batch_size * (2 if do_classifier_free_guidance else 1),
                                1,
                                1,
                                1,
                            ),
                            t,
                            encoder_hidden_states=text_embeddings,
                            return_dict=False,
                        )
                        self.reference_control_reader.update(
                            self.reference_control_writer
                        )
                        noise_pred = self.denoising_unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings,
                            timestep_cond=context_frame_encodings,
                            return_dict=False,
                        )[0]
                        self.reference_control_reader.clear()
                        self.reference_control_writer.clear()
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )
                        # remove ref image noise pred and context noise pred
                        noise_pred = noise_pred[:, :, mask:, ...]

                        denoised_input_latents = self.scheduler.step(
                            noise_pred,
                            t,
                            input_latents[:, :, mask:, :, :],
                            **extra_step_kwargs,
                        ).prev_sample

                        input_latents[:, :, mask:, :, :] = denoised_input_latents

                        timestep_progress_bar.set_description(
                            f"Denoising Step: {i + 1}/{len(timesteps)}"
                        )
                        timestep_progress_bar.update(1)

                        if i == len(timesteps) - 1:
                            latents[:, :, context, ...] = input_latents.cpu()

                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

                context_progress_bar.set_description(
                    f"Context: {context_idx + 1}/{len(context_queue)}"
                )
                context_progress_bar.update(1)

        video_tensor = self.decode_latents(latents)
        return video_tensor
