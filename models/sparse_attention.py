import math

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from torch import einsum, nn
from torch.nn import Module

from .rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

# constant

TOKEN_SELF_ATTN_VALUE = -5e4

# helper functions


def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def to(t):
    return {"device": t.device, "dtype": t.dtype}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim=-1)
    return normed.type(dtype)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


# main class


class LocalAttention(Module):
    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        dropout=0.0,
        shared_qk=False,
        rel_pos_emb_config=None,
        dim=None,
        autopad=False,
        exact_windowsize=False,
        scale=None,
        use_rotary_pos_emb=True,
        use_xpos=False,
        xpos_scale_base=None,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), "you cannot look forward if causal"

        self.scale = scale

        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize

        self.causal = causal

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = nn.Dropout(dropout)

        self.shared_qk = shared_qk

        # relative positions

        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (
            exists(rel_pos_emb_config) or exists(dim)
        ):  # backwards compatible with old `rel_pos_emb_config` deprecated argument
            if exists(rel_pos_emb_config):
                dim = rel_pos_emb_config[0]

            self.rel_pos = SinusoidalEmbeddings(
                dim,
                use_xpos=use_xpos,
                scale_base=default(xpos_scale_base, window_size // 2),
            )

    def forward(
        self, q, k, v, mask=None, input_mask=None, attn_bias=None, window_size=None
    ):
        dtype = q.dtype
        mask = default(mask, input_mask)

        assert not (
            exists(window_size) and not self.use_xpos
        ), "cannot perform window size extrapolation if xpos is not turned on"

        (
            shape,
            autopad,
            pad_value,
            window_size,
            causal,
            look_backward,
            look_forward,
            shared_qk,
        ) = (
            q.shape,
            self.autopad,
            -1,
            default(window_size, self.window_size),
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], "* n d"), (q, k, v))

        # auto padding

        if autopad:
            orig_seq_len = q.shape[1]
            (needed_pad, q), (_, k), (_, v) = map(
                lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v)
            )

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        scale = default(self.scale, dim_head**-0.5)

        assert (
            n % window_size
        ) == 0, f"sequence length {n} must be divisible by window size {window_size} for local attention"

        windows = n // window_size

        if shared_qk:
            k = l2norm(k)

        seq = torch.arange(n, device=device)
        b_t = rearrange(seq, "(w n) -> 1 w n", w=windows, n=window_size)

        # bucketing

        bq, bk, bv = map(
            lambda t: rearrange(t, "b (w n) d -> b w n d", w=windows), (q, k, v)
        )

        bq = bq * scale

        look_around_kwargs = dict(
            backward=look_backward, forward=look_forward, pad_value=pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        # rotary embeddings

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)

        # calculate positions for masking

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, "... i -> ... i 1")
        bq_k = rearrange(bq_k, "... j -> ... 1 j")

        pad_mask = bq_k == pad_value

        sim = einsum("b h i e, b h j e -> b h i j", bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0

            attn_bias = repeat(attn_bias, "h i j -> (b h) 1 i j", b=b // heads)
            sim = sim + attn_bias
        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, TOKEN_SELF_ATTN_VALUE)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k

            if self.exact_windowsize:
                max_causal_window_size = self.window_size * self.look_backward
                causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        # masking out for exact window size for non-causal
        # as well as masking out for padding value

        if not causal and self.exact_windowsize:
            max_backward_window_size = self.window_size * self.look_backward
            max_forward_window_size = self.window_size * self.look_forward
            window_mask = (
                ((bq_k - max_forward_window_size) > bq_t)
                | (bq_t > (bq_k + max_backward_window_size))
                | pad_mask
            )
            sim = sim.masked_fill(window_mask, mask_value)
        else:
            sim = sim.masked_fill(pad_mask, mask_value)

        # take care of key padding mask passed in

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0

            h = b // mask.shape[0]

            if autopad:
                _, mask = pad_to_multiple(mask, window_size, dim=-1, value=False)

            mask = rearrange(mask, "... (w n) -> (...) w n", w=windows, n=window_size)
            mask = look_around(mask, **{**look_around_kwargs, "pad_value": False})
            mask = rearrange(mask, "... j -> ... 1 j")
            mask = repeat(mask, "b ... -> (b h) ...", h=h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        attn = attn.to(dtype)
        # aggregation

        out = einsum("b h i j, b h j e -> b h i e", attn, bv)
        out = rearrange(out, "b w n d -> b (w n) d")

        if autopad:
            out = out[:, :orig_seq_len, :]

        out, *_ = unpack(out, packed_shape, "* n d")
        return out


class GlobalAndLocalMHA(Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        prenorm=False,
        qk_rmsnorm=True,
        qk_scale=8,
        use_xpos=False,
        xpos_scale_base=None,
        exact_windowsize=None,
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)
        self.norm = nn.LayerNorm(dim) if prenorm else None

        # Shared QKV projections for both global and local attention
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.causal = causal

        # Local attention for rest of sequence
        self.local_attn = LocalAttention(
            dim=dim_head,
            window_size=window_size,
            causal=causal,
            autopad=True,
            dropout=dropout,
            scale=(qk_scale if qk_rmsnorm else None),
            exact_windowsize=exact_windowsize,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(self, x, mask=None, attn_bias=None, cache=None, return_cache=False):
        dtype = x.dtype
        # Case 1: First token (clip feature, global token) - no cache
        if not exists(cache) and x.shape[1] == 1:
            # Process global token using shared projections
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
            )

            if self.qk_rmsnorm:
                q, k = map(l2norm, (q, k))
                q = q * self.q_scale
                k = k * self.k_scale

            # Global self-attention
            global_sim = einsum("b h i d, b h j d -> b h i j", q, k)
            global_attn = global_sim.softmax(dim=-1)
            global_out = einsum("b h i j, b h j d -> b h i d", global_attn, v)
            global_out = rearrange(global_out, "b h n d -> b n (h d)")
            out = self.to_out(global_out)

            if return_cache:
                kv_cache = torch.stack((k, v))
                return out, kv_cache
            return out

        # Case 2: New token with cache
        elif exists(cache):
            # Get cached KV pairs
            ck, cv = cache
            # First cached KV pair is global token
            global_k = ck[:, :, 0:1]  # [B, H, 1, D]
            global_v = cv[:, :, 0:1]  # [B, H, 1, D]
            cached_k = ck[:, :, 1:]   # Local cached keys
            cached_v = cv[:, :, 1:]   # Local cached values

            # Process new token
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

            if self.qk_rmsnorm:
                q, k = map(l2norm, (q, k))
                q = q * self.q_scale
                k = k * self.k_scale

            # Cross attention with global token
            cross_attn = einsum("b h n d, b h m d -> b h n m", q, global_k)
            cross_attn = cross_attn.softmax(dim=-1)
            global_context = einsum("b h n m, b h m d -> b h n d", cross_attn, global_v)

            # Local attention - similar to LocalMHA cached computation
            k = torch.cat((cached_k, k), dim=-2)
            v = torch.cat((cached_v, v), dim=-2)

            # Handle rotary embeddings if present
            if exists(self.local_attn.rel_pos):
                pos_emb, xpos_scale = self.local_attn.rel_pos(k)
                q, k = apply_rotary_pos_emb(q, k, pos_emb, scale=xpos_scale)

            # Window size management for local attention
            effective_window_size = self.local_attn.look_backward * self.window_size
            if self.exact_windowsize:
                kv_start_index = -(effective_window_size + 1)
            else:
                seq_len = k.shape[-2]
                kv_start_index = -(effective_window_size + (seq_len % self.window_size))
            k, v = tuple(t[..., kv_start_index:, :] for t in (k, v))

            # Direct attention computation
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            q = q * (q.shape[-1] ** -0.5)  # Scale query
            sim = einsum("b h i d, b h j d -> b h i j", q, k)

            if exists(attn_bias):
                k_len = k.shape[-2]
                attn_bias = attn_bias[..., -1:, -k_len:]
                sim = sim + attn_bias

            # Local attention
            attn = sim.softmax(dim=-1)
            local_out = einsum("b h i j, b h j d -> b h i d", attn, v)
            # Combine global and local attention
            combined_out = local_out + global_context
            combined_out = rearrange(combined_out, "b h n d -> b n (h d)")
            out = self.to_out(combined_out)

            if return_cache:
                # Include global token in cache
                k = torch.cat([global_k, k], dim=2)
                v = torch.cat([global_v, v], dim=2)
                kv_cache = torch.stack((k, v))
                return out, kv_cache
            return out

        # Case 3: Training with global token + local tokens
        else:
            global_token, local_seq = x[:, 0:1], x[:, 1:]

            # Process global token
            gq, gk, gv = self.to_qkv(global_token).chunk(3, dim=-1)
            gq, gk, gv = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (gq, gk, gv)
            )

            if self.qk_rmsnorm:
                gq, gk = map(l2norm, (gq, gk))
                gq = gq * self.q_scale
                gk = gk * self.k_scale

            # Global self-attention
            global_sim = einsum("b h i d, b h j d -> b h i j", gq, gk)
            global_attn = global_sim.softmax(dim=-1)
            global_out = einsum("b h i j, b h j d -> b h i d", global_attn, gv)
            global_out = rearrange(global_out, "b h n d -> b n (h d)")
            global_out = self.to_out(global_out)

            # Process local sequence
            q, k, v = self.to_qkv(local_seq).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
            )

            if self.qk_rmsnorm:
                q, k = map(l2norm, (q, k))
                q = q * self.q_scale
                k = k * self.k_scale

            # Cross attention with global token
            cross_attn = einsum("b h n d, b h m d -> b h n m", q, gk)
            cross_attn = cross_attn.softmax(dim=-1)
            global_context = einsum("b h n m, b h m d -> b h n d", cross_attn, gv)

            # Local attention
            local_out = self.local_attn(q, k, v, mask=mask, attn_bias=attn_bias)
            # Combine global and local attention
            combined_out = local_out + global_context
            combined_out = rearrange(combined_out, "b h n d -> b n (h d)")
            local_out = self.to_out(combined_out)

            # Combine global and local outputs
            out = torch.cat([global_out, local_out], dim=1)

            if return_cache:
                k = torch.cat([gk, k], dim=2)
                v = torch.cat([gv, v], dim=2)
                kv_cache = torch.stack((k, v))
                return out, kv_cache
            return out


if __name__ == "__main__":
    q = torch.randn(2, 8, 2048, 64)
    k = torch.randn(2, 8, 2048, 64)
    v = torch.randn(2, 8, 2048, 64)

    attn = LocalAttention(
        dim=64,  # dimension of each head (you need to pass this in for relative positional encoding)
        window_size=512,  # window size. 512 is optimal, but 256 or 128 yields good enough results
        causal=True,  # auto-regressive or not
        look_backward=1,  # each window looks at the window before
        look_forward=0,  # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
        dropout=0.1,  # post-attention dropout
        exact_windowsize=False,  # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
        global_tokens=1,
    )

    # mask = torch.ones(2, 2048).bool()
    out = attn(q, k, v)  # (2, 8, 2048, 64)
