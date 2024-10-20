import torch
import torch.nn as nn
from .utils import apply_rotary_emb, precompute_freqs_cis

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.key = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.value = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, S, D = x.shape
        query = self.query(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        key = self.key(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value(x).view(B, S, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        dots = (query @ key.transpose(-1, -2)) * self.scale


        mask = torch.tril(torch.ones((S, S))).to(x.device)
        dots.masked_fill_(mask == 0, float('-inf'))

        att_scores = dots.softmax(-1)
        att_v = att_scores @ value

        out = att_v.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        out = self.out(out)

        return out


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model is not divisible by h"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.key = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.value = nn.Linear(d_model, d_model, bias = qkv_bias)
        self.out = nn.Linear(d_model, d_model)
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B, S, D = x.shape
        query = self.query(x).view(B, S, self.n_heads, self.d_head)
        key = self.key(x).view(B, S, self.n_heads, self.d_head)
        value = self.value(x).view(B, S, self.n_heads, self.d_head).transpose(1,2)
        query, key = apply_rotary_emb(query, key, precompute_freqs_cis(self.d_head, 4096)[:S].to(next(iter(self.query.parameters())).device))
        query = query.transpose(1,2)
        key = key.transpose(1,2)

        dots = (query @ key.transpose(-1, -2)) * self.scale


        mask = torch.tril(torch.ones((S, S))).to(x.device)
        dots.masked_fill_(mask == 0, float('-inf'))

        att_scores = dots.softmax(-1)
        att_v = att_scores @ value

        out = att_v.permute(0, 2, 1, 3).contiguous().view(B, S, D)

        out = self.out(out)

        return out


