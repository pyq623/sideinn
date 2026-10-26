import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization used for the compressed KV latent."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


def apply_rotary_embedding(x, base=10_000.0, position_offset=0):
    """Apply RoPE to pairs in the last dimension of [B, H, S, R]."""
    rope_dim = x.size(-1)
    if rope_dim == 0:
        return x
    if rope_dim % 2 != 0:
        raise ValueError("qk_rope_head_dim must be even")

    seq_len = x.size(-2)
    positions = torch.arange(
        position_offset,
        position_offset + seq_len,
        device=x.device,
        dtype=torch.float32,
    )
    dimensions = torch.arange(0, rope_dim, 2, device=x.device).float()
    inverse_frequencies = base ** (-dimensions / rope_dim)
    angles = torch.outer(positions, inverse_frequencies)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    dtype = x.dtype
    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        dim=-1,
    ).flatten(-2)
    return rotated.to(dtype)


class MultiHeadLatentAttention(nn.Module):
    """Interview-oriented DeepSeek-style Multi-Head Latent Attention."""

    def __init__(
        self,
        d_model,
        num_heads,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        value_head_dim,
        dropout=0.0,
        rope_base=10_000.0,
    ):
        super().__init__()

        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive")
        if qk_nope_head_dim < 0 or qk_rope_head_dim < 0:
            raise ValueError("Q/K head dimensions cannot be negative")
        if qk_nope_head_dim + qk_rope_head_dim == 0:
            raise ValueError("Q/K head dimension must be positive")
        if qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")
        if value_head_dim <= 0:
            raise ValueError("value_head_dim must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.value_head_dim = value_head_dim
        self.rope_base = rope_base

        # Q remains multi-head, then splits into non-positional and RoPE parts.
        self.q_proj = nn.Linear(d_model, num_heads * self.qk_head_dim)

        # The down projection emits one shared KV latent plus one shared RoPE K.
        self.kv_down_proj = nn.Linear(
            d_model,
            kv_lora_rank + qk_rope_head_dim,
        )
        self.kv_norm = RMSNorm(kv_lora_rank)

        # The latent is expanded into each head's non-positional K and V.
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + value_head_dim),
        )
        self.out_proj = nn.Linear(num_heads * value_head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=False, attention_mask=None):
        B, S, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected input dimension {self.d_model}, got {D}")

        # Q: [B, S, H * (D_nope + D_rope)] -> [B, H, S, Dqk]
        q = self.q_proj(x)
        q = q.view(B, S, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_embedding(q_rope, self.rope_base)

        # Compress x into the latent cached by MLA and a shared positional key.
        compressed = self.kv_down_proj(x)
        kv_latent, k_rope = compressed.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        # Expand only for the straightforward training/interview computation.
        kv = self.kv_up_proj(self.kv_norm(kv_latent))
        kv = kv.view(
            B,
            S,
            self.num_heads,
            self.qk_nope_head_dim + self.value_head_dim,
        ).transpose(1, 2)
        k_nope, v = kv.split(
            [self.qk_nope_head_dim, self.value_head_dim],
            dim=-1,
        )

        # One positional K is shared by all heads; expand creates only a view.
        k_rope = apply_rotary_embedding(k_rope.unsqueeze(1), self.rope_base)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        q = torch.cat((q_nope, q_rope), dim=-1)
        k = torch.cat((k_nope, k_rope), dim=-1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.qk_head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(
            B,
            S,
            self.num_heads * self.value_head_dim,
        )
        return self.out_proj(out)
