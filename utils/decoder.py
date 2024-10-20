from .attention import MultiHeadAttention, MultiHeadAttentionRoPE
import torch
import torch.nn as nn
from torchtune.modules import RMSNorm
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.MHA = MultiHeadAttention(d_model, n_heads, qkv_bias)
        self.MLP = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):

        x = self.layer_norm_1(self.MHA(x)) + x
        x = self.layer_norm_2(self.MLP(x)) + x

        return x
    
class FeedForwardllama(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)
        self.w3 = nn.Linear(d_model, 4 * d_model)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DecoderBlockllama(nn.Module):
    def __init__(self, d_model, n_heads, qkv_bias = True):
        super().__init__()
        self.RMSNorm_1 = RMSNorm(d_model)
        self.RMSNorm_2 = RMSNorm(d_model)

        self.MHA = MultiHeadAttentionRoPE(d_model, n_heads, qkv_bias)
        self.MLP = FeedForwardllama(d_model)

    def forward(self, x):

        x = self.MHA(self.RMSNorm_1(x)) + x
        x = self.MLP(self.RMSNorm_2(x)) + x

        return x
