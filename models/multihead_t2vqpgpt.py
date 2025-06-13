import math

import models.pos_encoding as pos_encoding
import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions import Categorical
from torch.nn import functional as F


class Text2VQPoseGPT(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
        attn_type="default",
        head_type="default",
        pose_size=10,
        head_layers=None,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.pose_size = pose_size
        self.head_type = head_type
        print(f"use headtype:{head_type}")
        self.trans_base = CrossCondTransBase(
            num_vq,
            embed_dim,
            clip_dim,
            block_size,
            num_layers,
            n_head,
            drop_out_rate,
            fc_rate,
            attn_type,
            pose_size,
        )

        # Multiple decoding heads for each position in pose
        self.trans_heads = nn.ModuleList(
            [
                (
                    CrossCondTransHead(
                        num_vq,
                        embed_dim,
                        block_size,
                        head_layers if head_layers is not None else num_layers,
                        n_head,
                        drop_out_rate,
                        fc_rate,
                    )
                    if self.head_type == "default"
                    else MLPTransHead(
                        num_vq,
                        embed_dim,
                        block_size,
                        head_layers if head_layers is not None else num_layers,
                        n_head,
                        drop_out_rate,
                        fc_rate,
                    )
                )
                for _ in range(pose_size)
            ]
        )

        self.block_size = block_size
        self.num_vq = num_vq

    def clear_cache(self):
        self.trans_base.clear_cache()
        for head in self.trans_heads:
            head.clear_cache()

    def forward(self, idxs, clip_feature, flat=True, use_cache=False, shift_t=None):
        weight_type = self.trans_base.tok_emb.weight.dtype
        clip_feature = clip_feature.to(weight_type)
        feat = self.trans_base(idxs, clip_feature, use_cache=use_cache, shift_t=shift_t)

        # Get predictions from all heads
        all_logits = []
        for head in self.trans_heads:
            logits = head(feat, use_cache=use_cache)
            all_logits.append(logits)

        # Stack predictions along a new dimension
        logits = torch.stack(all_logits, dim=1)  # [batch, pose_size, seq_len, num_vq+1]
        logits = rearrange(logits, "b p s d -> b s p d")
        if flat:
            logits = rearrange(logits, "b s p d -> b (s p) d", p=self.pose_size)
        return logits

    def sample(self, clip_feature, max_pose_len=None, if_categorial=False):
        max_pose_len = max_pose_len or self.block_size

        # Clear cache at the start of sampling
        self.clear_cache()

        # Initialize with empty sequence
        xs = None

        for k in range(max_pose_len):
            if k == 0:
                # First step: empty sequence
                x = []
            else:
                # Use only the last generated tokens
                x = xs[:, -1:]  # [batch, 1, pose_size]
                x = rearrange(x, "b t p -> b (t p)")

            # Use cache during sampling
            logits = self.forward(
                x, clip_feature, flat=False, use_cache=True, shift_t=k
            )
            logits = logits[:, -1]  # Get predictions for next position

            # Process predictions for all pose positions
            probs = F.softmax(logits, dim=-1)

            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                idx = idx.squeeze(-1)

            # Append to sequence
            if k == 0:
                xs = idx.unsqueeze(1)
            else:
                xs = torch.cat((xs, idx.unsqueeze(1)), dim=1)

        xs = rearrange(xs, "b s p -> b (s p)", p=self.pose_size)
        return xs


class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        block_size=16,
        n_head=8,
        drop_out_rate=0.1,
        attn_type="default",
    ):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.cached_k = None
        self.cached_v = None

    def clear_cache(self):
        self.cached_k = None
        self.cached_v = None

    def forward(self, x, use_cache=False):
        B, T, C = x.size()

        # calculate query, key, values
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if use_cache:
            if self.cached_k is not None:
                k = torch.cat([self.cached_k, k], dim=2)
                v = torch.cat([self.cached_v, v], dim=2)
            self.cached_k = k
            self.cached_v = v

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=not use_cache,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        block_size=16,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
        attn_type="default",
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(
            embed_dim, block_size, n_head, drop_out_rate, attn_type
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, use_cache=False):
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossCondTransBase(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        clip_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
        attn_type="default",
        pose_size=10,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)

        self.token_combiner = nn.Sequential(
            nn.Linear(embed_dim * pose_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.pose_size = pose_size

        # transformer block
        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate, attn_type)
                for _ in range(num_layers)
            ]
        )
        self.pos_embed = pos_encoding.PositionEmbedding(
            block_size, embed_dim, 0.0, False
        )

        self.block_size = block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    def forward(self, idx, clip_feature, use_cache=False, shift_t=None):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
            positions = 0  # Start position for empty sequence
        else:
            b = idx.shape[0]
            idx_reshaped = idx.view(b, -1, self.pose_size)
            idx_reshaped = idx_reshaped.to(dtype=torch.int64)
            b, t, p = idx_reshaped.shape
            assert (
                t <= self.block_size
            ), "Cannot forward, model block size is exhausted."

            # Get current sequence length for proper positional encoding
            positions = (
                shift_t if use_cache else 0
            )  # If using cache, only encode new position

            token_embeddings_list = []
            for i in range(idx_reshaped.size(1)):
                pose_tokens = idx_reshaped[:, i]
                pose_embeds = self.tok_emb(pose_tokens)
                pose_embeds_flat = pose_embeds.view(b, -1)
                combined_embed = self.token_combiner(pose_embeds_flat)
                token_embeddings_list.append(combined_embed)

            token_embeddings = torch.stack(token_embeddings_list, dim=1)

            token_embeddings = torch.cat(
                [self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1
            )

            if use_cache:
                # When using cache, only process the new tokens
                token_embeddings = token_embeddings[:, -1:]

        # Apply position embeddings with correct offset
        x = self.pos_embed(token_embeddings, offset=positions)

        for block in self.blocks:
            x = block(x, use_cache=use_cache)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def clear_cache(self):
        for block in self.blocks:
            block.clear_cache()

    def forward(self, x, use_cache=False):
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class MLPTransHead(nn.Module):

    def __init__(
        self,
        num_vq=1024,
        embed_dim=512,
        block_size=16,
        num_layers=2,
        n_head=8,
        drop_out_rate=0.1,
        fc_rate=4,
    ):
        super().__init__()

        # self.blocks = nn.Sequential(
        #     *[
        #         Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate)
        #         for _ in range(num_layers)
        #     ]
        # )
        self.block = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def clear_cache(self):
        pass

    def forward(self, x, use_cache=False):
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
