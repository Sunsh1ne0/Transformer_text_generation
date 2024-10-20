from .encoder import PositionalEncoding
from.decoder import DecoderBlock, DecoderBlockllama
import torch
import torch.nn as nn
from torchtune.modules import RMSNorm

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_blocks, qkv_bias = True, seq_len = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, seq_len)
        self.blocks = nn.Sequential(*[DecoderBlock(d_model, n_heads, qkv_bias) for _ in range(n_blocks)])
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.to(next(iter(self.parameters())).device)
        emb = self.embedding(x)
        emb = self.pos(emb)

        x = self.blocks(emb)

        out = self.final_layer(x)

        return out
    
class TransformerDecoderllama(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_blocks, qkv_bias = True, seq_len = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.Sequential(*[DecoderBlockllama(d_model, n_heads, qkv_bias) for _ in range(n_blocks)])
        self.RMSNorn = RMSNorm(d_model)
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.to(next(iter(self.parameters())).device)
        emb = self.embedding(x)

        x = self.blocks(emb)
        x = self.RMSNorn(x)
        out = self.final_layer(x)

        return out