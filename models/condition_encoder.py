from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.controlnet import zero_module
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.nn import functional as F

from .motion_module import get_motion_module
from .vq.basic_vae import Decoder, Encoder
from .vq.FSQ import FSQ
from .vq.LFQ import LFQ
from .vq.residual_vq import ResidualVQ
from .vq.vector_quantize_pytorch import VectorQuantize


class MultiCondBackbone(nn.Module):
    def __init__(
        self,
        conditioning_channels: int = 2,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        num_conds: int = 2,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels * num_conds,
            block_out_channels[0] * num_conds,
            kernel_size=(3, 3),
            groups=num_conds,
            padding=1,
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(
                    channel_in * num_conds,
                    channel_in * num_conds,
                    kernel_size=(3, 3),
                    groups=num_conds,
                    padding=1,
                )
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in * num_conds,
                    channel_out * num_conds,
                    kernel_size=(3, 3),
                    groups=num_conds,
                    padding=1,
                    stride=(2, 2),
                )
            )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        return embedding


class GateModule(nn.Module):
    def __init__(self, channels: int, num_conds: int = 2):
        super(GateModule, self).__init__()
        self.channels = channels
        self.num_conds = num_conds
        self.gate_layer = nn.Sequential(
            nn.Conv2d(
                self.channels * self.num_conds,
                self.channels // 2,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                self.channels // 2, self.num_conds, kernel_size=(7, 7), padding=3
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        gate_weight = self.gate_layer(x).reshape(B, self.num_conds, 1, H, W)
        x = x.reshape(B, self.num_conds, -1, H, W)
        x = x * gate_weight
        x = x.reshape(B, C, H, W)
        return x


class ConditionEncoder(ModelMixin):
    def __init__(
        self,
        conditioning_channels: int = 3,
        backbone_channels: Tuple[int, ...] = (16, 32, 96, 256),
        out_channels: Tuple[int, ...] = (320, 320, 640, 1280, 1280),
        image_finetune: bool = False,
        motion_module_type: Optional = None,
        motion_module_kwargs: Optional = None,
        num_conds: int = 2,
    ):
        super(ConditionEncoder, self).__init__()
        self.conditioning_channels = conditioning_channels
        self.backbone_channels = backbone_channels
        self.out_channels = out_channels
        self.image_finetune = image_finetune
        self.num_conds = num_conds

        self.backbone = MultiCondBackbone(
            conditioning_channels=self.conditioning_channels,
            block_out_channels=self.backbone_channels,
            num_conds=self.num_conds,
        )

        self.gate_module = GateModule(
            channels=backbone_channels[-1], num_conds=num_conds
        )

        self.blocks_0 = nn.Sequential(
            nn.Conv2d(
                backbone_channels[-1] * num_conds, out_channels[0], kernel_size=(1, 1)
            ),
            nn.SiLU(),
        )
        self.block_0_out_proj = zero_module(
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=(1, 1))
        )

        motion_modules = []
        if not self.image_finetune:
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels[0],
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                )
            )

        for i in range(1, len(out_channels)):
            self.register_module(
                f"blocks_{i}",
                nn.Sequential(
                    nn.Conv2d(
                        out_channels[i - 1],
                        out_channels[i],
                        kernel_size=(3, 3),
                        padding=1,
                        stride=(2, 2) if i < len(out_channels) - 1 else (1, 1),
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels[i],
                        out_channels[i],
                        kernel_size=(3, 3),
                        padding=1,
                        stride=(1, 1),
                    ),
                    nn.SiLU(),
                ),
            )
            self.register_module(
                f"block_{i}_out_proj",
                zero_module(
                    nn.Conv2d(out_channels[i], out_channels[i], kernel_size=(1, 1))
                ),
            )
            if not self.image_finetune:
                motion_modules.append(
                    get_motion_module(
                        in_channels=out_channels[i],
                        motion_module_type=motion_module_type,
                        motion_module_kwargs=motion_module_kwargs,
                    )
                )

        if not self.image_finetune:
            self.motion_modules = nn.ModuleList(motion_modules)

    def encode(
        self,
        dwpose,
        hamer,
    ):
        cond = torch.cat((dwpose, hamer), dim=1)
        cond = self.backbone(cond)
        cond = self.gate_module(cond)
        return cond

    def decode(self, cond, video_length=-1, temb=None, encoder_hidden_states=None):
        outs = []
        for i in range(len(self.out_channels)):
            cond = self.get_submodule(f"blocks_{i}")(cond)
            if not self.image_finetune and video_length > 1:
                cond = rearrange(cond, "(b f) c h w -> b c f h w", f=video_length)
                cond = self.motion_modules[i](
                    cond, temb=temb, encoder_hidden_states=encoder_hidden_states
                )
                cond = rearrange(cond, "b c f h w -> (b f) c h w")
            outs.append(self.get_submodule(f"block_{i}_out_proj")(cond))

        return tuple(outs)

    def forward(
        self,
        dwpose,
        hamer,
        temb=None,
        encoder_hidden_states=None,
        video_length=-1,
        return_cond=False,
    ):
        cond = torch.cat((dwpose, hamer), dim=1)
        cond = self.backbone(cond)
        cond = self.gate_module(cond)
        mid_cond = cond
        outs = []

        for i in range(len(self.out_channels)):
            cond = self.get_submodule(f"blocks_{i}")(cond)
            if not self.image_finetune and video_length > 1:
                cond = rearrange(cond, "(b f) c h w -> b c f h w", f=video_length)
                cond = self.motion_modules[i](
                    cond, temb=temb, encoder_hidden_states=encoder_hidden_states
                )
                cond = rearrange(cond, "b c f h w -> (b f) c h w")
            outs.append(self.get_submodule(f"block_{i}_out_proj")(cond))

        if return_cond:
            return tuple(outs), mid_cond
        else:
            return tuple(outs)


class VQModel(nn.Module):
    def __init__(
        self,
        vq_type: str,
        n_e: int,
        in_channels: int,
        quantizer_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        input_size: Tuple[int, int] = (34, 28),
        skip_vq: bool = False,
        fsq_levels: Tuple[int, ...] = (8, 5, 5, 5),
        **kwargs,
    ):
        super().__init__()
        self.vq_type = vq_type
        self.n_e = n_e
        self.skip_vq = skip_vq
        self.downsample_encoder = Encoder(
            z_channels=quantizer_channels,
            in_channels=in_channels,
            ch_mult=ch_mult,
            input_size=input_size,
        )

        if self.vq_type == "default":
            self.quantizer = VectorQuantize(
                dim=quantizer_channels,
                codebook_size=self.n_e,
                kmeans_init=True,
                sync_codebook=False,
                codebook_dim=8,
                **kwargs,
            )
        elif self.vq_type == "LFQ":
            self.quantizer = LFQ(
                dim=quantizer_channels,
                codebook_size=self.n_e,
                entropy_loss_weight=0.1,
                diversity_gamma=1.0,
                experimental_softplus_entropy_loss=True,
                **kwargs,
            )
        elif self.vq_type == "residual_vq":
            self.quantizer = ResidualVQ(
                dim=quantizer_channels,
                codebook_size=self.n_e,
                codebook_dim=8,
                kmeans_init=True,
                sync_codebook=False,
                shared_codebook=True,
                **kwargs,
            )
        elif self.vq_type == "FSQ":
            self.quantizer = FSQ(
                levels=fsq_levels,
                dim=quantizer_channels,
            )
        self.upsample_decoder = Decoder(
            z_channels=quantizer_channels,
            in_channels=in_channels,
            ch_mult=ch_mult,
            output_size=self.downsample_encoder.feature_sizes,
        )

    def forward(self, x, inner_cond=False):
        x = self.downsample_encoder(x)
        bf, c, h, w = x.shape
        x = rearrange(x, "bf c h w -> bf (h w) c")
        if self.skip_vq:
            z_q = x
            indices = torch.arange(self.n_e, device=x.device)
            embedding_loss = torch.tensor([0.0], device=x.device)
        else:
            if self.vq_type == "FSQ":
                z_q, indices = self.quantizer(x)
                embedding_loss = torch.tensor([0.0], device=x.device)
            else:
                z_q, indices, embedding_loss = self.quantizer(x)
        z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h, w=w)
        indices_count = torch.bincount(indices.view(-1), minlength=self.n_e)

        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.all_reduce(indices_count)
        avg_probs = indices_count.float() / indices_count.sum()

        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()

        if inner_cond:
            return z_q, perplexity, embedding_loss
        x = self.upsample_decoder(z_q)

        return x, perplexity, embedding_loss

    def encode(self, x):
        x = self.downsample_encoder(x)
        bf, c, h, w = x.shape
        if self.skip_vq:
            return x
        x = rearrange(x, "bf c h w -> bf (h w) c")
        if self.vq_type == "FSQ":
            z_q, indices = self.quantizer(x)
            embedding_loss = torch.tensor([0.0], device=x.device)
        else:
            z_q, indices, embedding_loss = self.quantizer(x)
        return indices

    def decode(self, z):
        weight_dtype = self.downsample_encoder.conv_in.weight.dtype
        h, w = self.downsample_encoder.feature_sizes[-1]
        # print(f"h: {h}, w: {w}")
        # print(f"z: {z.shape}")
        if self.skip_vq:
            z_q = z
            z_q = rearrange(z_q, "b (h w c) -> b c h w", h=h, w=w)
        else:
            z = z.to(torch.long)
            if self.vq_type == "FSQ":
                z_q = self.quantizer.indices_to_codes(z)
                z_q = z_q.to(weight_dtype)
            else:
                z_q = self.quantizer.get_output_from_indices(z)

            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h, w=w)

        cond = self.upsample_decoder(z_q)

        return cond


class VQConditionEncoder(ConditionEncoder):
    def __init__(
        self,
        conditioning_channels: int = 3,
        backbone_channels: Tuple[int, ...] = (16, 32, 96, 256),
        out_channels: Tuple[int, ...] = (320, 320, 640, 1280, 1280),
        image_finetune: bool = False,
        motion_module_type: Optional = None,
        motion_module_kwargs: Optional = None,
        num_conds: int = 2,
        num_compress: int = 1,
        channel_reduction_factor: int = 1,
        use_vq: bool = False,
        vq_kwargs: Optional = None,
    ):
        super().__init__(
            conditioning_channels=conditioning_channels,
            backbone_channels=backbone_channels,
            out_channels=out_channels,
            image_finetune=image_finetune,
            motion_module_type=motion_module_type,
            motion_module_kwargs=motion_module_kwargs,
            num_conds=num_conds,
        )
        self.num_compress = num_compress
        self.channel_reduction_factor = channel_reduction_factor
        self.use_vq = use_vq

        orig_channels = backbone_channels[-1] * self.num_conds
        compressed_channels = orig_channels // channel_reduction_factor

        if use_vq:
            vq_kwargs = vq_kwargs or {}
            self.vq = VQModel(in_channels=compressed_channels, **vq_kwargs)

    def encode(self, dwpose, hamer, return_indices=True, inner_cond=False):
        cond = torch.cat((dwpose, hamer), dim=1)
        cond = self.backbone(cond)
        cond = self.gate_module(cond)

        if self.use_vq:
            if return_indices:
                indices = self.vq.encode(cond)
                return indices
            else:
                cond, perplexity, embedding_loss = self.vq(cond, inner_cond=inner_cond)
                return cond, perplexity, embedding_loss
        return cond

    def decode(self, cond, video_length=-1, temb=None, encoder_hidden_states=None):
        outs = []
        for i in range(len(self.out_channels)):
            cond = self.get_submodule(f"blocks_{i}")(cond)
            if not self.image_finetune and video_length > 1:
                cond = rearrange(cond, "(b f) c h w -> b c f h w", f=video_length)
                cond = self.motion_modules[i](
                    cond, temb=temb, encoder_hidden_states=encoder_hidden_states
                )
                cond = rearrange(cond, "b c f h w -> (b f) c h w")
            outs.append(self.get_submodule(f"block_{i}_out_proj")(cond))

        return tuple(outs)

    def forward(
        self,
        dwpose,
        hamer,
        temb=None,
        encoder_hidden_states=None,
        video_length=-1,
        return_cond=False,
        return_vq=False,
    ):
        
        if self.use_vq:
            cond, perplexity, embedding_loss = self.encode(
                dwpose, hamer, return_indices=False
            )
        else:
            cond = self.encode(dwpose, hamer)

            mid_cond = cond
        outs = self.decode(cond, video_length, temb, encoder_hidden_states)
        
        if return_vq and self.use_vq:
            return outs, perplexity, embedding_loss
        
        if return_cond:
            return outs, mid_cond
        else:
            return outs

    def preprocess(self, z):
        cond = self.vq.decode(z)
        return cond

    def infer(self, cond, video_length=-1, temb=None, encoder_hidden_states=None):
        outs = []
        for i in range(len(self.out_channels)):
            cond = self.get_submodule(f"blocks_{i}")(cond)
            if not self.image_finetune and video_length > 1:
                cond = rearrange(cond, "(b f) c h w -> b c f h w", f=video_length)
                cond = self.motion_modules[i](
                    cond, temb=temb, encoder_hidden_states=encoder_hidden_states
                )
                cond = rearrange(cond, "b c f h w -> (b f) c h w")
            outs.append(self.get_submodule(f"block_{i}_out_proj")(cond))

        return tuple(outs)
