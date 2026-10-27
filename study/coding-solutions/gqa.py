# 基于站主已有练习实现；仅补充全 mask 行的处理。
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiGroupAttention(nn.Module):
    def __init__(self, d_model, num_heads, group, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0
        assert 1 <= group <= num_heads
        assert num_heads % group == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.group = group
        self.head_dim = d_model // num_heads

        self.dropout = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * group * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False, attention_mask=None):
        B, S, D = x.shape

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        q = q.view(B, S, self.num_heads, self.head_dim)

        kv = kv.view(B, S, 2, self.group, self.head_dim)
        k, v = kv.unbind(dim=2)

        kv_num_heads = self.num_heads // self.group

        k = k.repeat_interleave(kv_num_heads, dim=2)
        v = v.repeat_interleave(kv_num_heads, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, float("-inf"))

        # 补充：全 mask 行的注意力权重定义为零，避免 softmax(-inf) 的 NaN。
        all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_masked, 0.0)
        scores = F.softmax(scores, dim=-1)
        scores = scores.masked_fill(all_masked, 0.0)
        attn = self.dropout(scores)

        attn = attn @ v
        out = attn.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(out)
