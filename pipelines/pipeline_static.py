# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from typing import Callable, List, Optional, Union

import PIL
import PIL.Image
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

from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import ConditionEncoder
from models.mutual_self_attention import ReferenceAttentionControl

from .context import get_context_scheduler


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


class SignViPStaticPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        denoising_unet,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        empty_text_emb=None,
        tokenizer=None,
        text_encoder=None,
        appearance_encoder: AppearanceEncoderModel = None,
        condition_encoder: ConditionEncoder = None,
        max_sin_pe_len=1000,
    ):
        super().__init__()

        # Register only non-None modules
        modules_dict = {
            "vae": vae,
            "denoising_unet": denoising_unet,
            "scheduler": scheduler,
            "appearance_encoder": appearance_encoder,
            "condition_encoder": condition_encoder,
        }

        # Only add tokenizer and text_encoder if they are provided
        if tokenizer is not None:
            modules_dict["tokenizer"] = tokenizer
        if text_encoder is not None:
            modules_dict["text_encoder"] = text_encoder

        self.register_modules(**modules_dict)

        self.empty_text_emb = empty_text_emb

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.image_transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )
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

        models_to_offload = [self.vae, self.denoising_unet]
        if hasattr(self, "text_encoder"):
            models_to_offload.append(self.text_encoder)

        for cpu_offloaded_model in models_to_offload:
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
        # If empty_text_emb is provided and text_encoder is not, use only empty_text_emb
        if self.empty_text_emb is not None and not hasattr(self, "text_encoder"):
            # Use empty_text_emb for both conditional and unconditional guidance
            # For conditional guidance
            text_embeddings = self.empty_text_emb.to(device)
            if len(text_embeddings.shape) == 2:  # Add batch dimension if needed
                text_embeddings = text_embeddings.unsqueeze(0)

            batch_size = 1  # Default batch size when using only empty_text_emb

            # Duplicate text embeddings for each prompt
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
            text_embeddings = text_embeddings.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, duplicate the embeddings
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

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if self.empty_text_emb is not None:
                # Use the provided empty_text_emb for unconditional embeddings
                uncond_embeddings = self.empty_text_emb.to(device)
                # duplicate unconditional embeddings for each generation per prompt
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
                uncond_embeddings = uncond_embeddings.repeat(
                    1, num_videos_per_prompt, 1
                )
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
        reference_image_latents = self.vae.encode(reference_image).latent_dist.mean
        reference_image_latents = (
            self.vae.config.scaling_factor * reference_image_latents
        )
        return reference_image_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        reference_image=None,
        sk_image=None,
        width=None,
        height=None,
        num_inference_steps=None,
        guidance_scale=7.5,
        hamer_image=None,
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
        reference_control_writer=None,
        reference_control_reader=None,
        text_embeddings=None,
        **kwargs,
    ):
        # Default height and width to unet
        height = (
            height or self.denoising_unet.config.sample_size * self.vae_scale_factor
        )
        width = width or self.denoising_unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0
        if reference_control_writer is None:
            reference_control_writer = ReferenceAttentionControl(
                self.appearance_encoder,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="write",
                batch_size=batch_size,
                fusion_blocks="full",
            )

        if reference_control_reader is None:
            reference_control_reader = ReferenceAttentionControl(
                self.denoising_unet,
                do_classifier_free_guidance=do_classifier_free_guidance,
                mode="read",
                batch_size=batch_size,
                fusion_blocks="full",
            )

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
        video_length = 1
        if isinstance(sk_image, PIL.Image.Image):
            sk_image = self.image_transform(sk_image).to(
                dtype=self.dtype, device=self.device
            )  # [b, c, h, w]
        else:
            sk_image = sk_image.to(dtype=self.dtype, device=self.device)  # [b, c, h, w]
        if len(sk_image.shape) == 3:
            sk_image = sk_image.unsqueeze(0).unsqueeze(2)  # [b, c, 1, h, w]
        elif len(sk_image.shape) == 4:
            sk_image = sk_image.unsqueeze(2)  # [b, c, 1, h, w]
        elif len(sk_image.shape) == 5:
            video_length = sk_image.shape[2]

        if hamer_image is not None:
            if isinstance(hamer_image, PIL.Image.Image):
                hamer_image = self.image_transform(hamer_image).to(
                    dtype=self.dtype, device=self.device
                )  # [b, c, h, w]
            else:
                hamer_image = hamer_image.to(
                    dtype=self.dtype, device=self.device
                )  # [b, c, h, w]
            if len(hamer_image.shape) == 3:
                hamer_image = hamer_image.unsqueeze(0).unsqueeze(2)  # [b, c, 1, h, w]
            elif len(hamer_image.shape) == 4:
                hamer_image = hamer_image.unsqueeze(2)  # [b, c, 1, h, w]

        # prepare latents
        if text_embeddings is None:
            if prompt is None and self.empty_text_emb is None:
                raise ValueError("Either prompt or text_embeddings must be provided.")
            text_embeddings = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
        else:
            # User provided text_embeddings directly
            text_embeddings = text_embeddings.to(device=device)
            # For classifier free guidance, duplicate if not already done
            if do_classifier_free_guidance and text_embeddings.shape[0] == 1:
                text_embeddings = torch.cat([text_embeddings, text_embeddings])

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            self.dtype,
            torch.device("cpu"),
            generator,
        ).to(dtype=self.dtype, device=self.device)

        ref_image_latents = self.prepare_reference_latents(reference_image)

        sk_image = rearrange(sk_image, "b c f h w -> (b f) c h w")
        hamer_image = rearrange(hamer_image, "b c f h w -> (b f) c h w")
        sk_features = self.condition_encoder(sk_image, hamer_image)
        sk_features = [
            rearrange(feature, "(b f) c h w -> b c f h w", b=batch_size)
            for feature in sk_features
        ]
        sk_features = [
            feature.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
            for feature in sk_features
        ]

        with self.progress_bar(total=len(timesteps)) as timestep_progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = latents  # [bs, c, f, w, h]
                latent_model_input = latent_model_input.repeat(
                    2 if do_classifier_free_guidance else 1, 1, 1, 1, 1
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
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
                reference_control_reader.update(
                    reference_control_writer, dtype=self.condition_encoder.dtype
                )
                noise_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    sk_features=sk_features,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                ).prev_sample

                reference_control_reader.clear()
                reference_control_writer.clear()

                timestep_progress_bar.set_description(
                    f"Denoising Step: {i + 1}/{len(timesteps)}"
                )
                timestep_progress_bar.update(1)

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        reference_control_reader.restore_hacked_forward()
        reference_control_writer.restore_hacked_forward()
        video_tensor = self.decode_latents(latents)
        return video_tensor
