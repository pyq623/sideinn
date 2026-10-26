import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False, attention_mask=None):
        B, S, D = x.shape
        qkv = self.qkv_proj(x)

        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-1,-2)) / math.sqrt(self.head_dim)

        if causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~mask, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        attn = self.dropout(scores)

        attn = attn @ v

        attn = attn.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(attn)