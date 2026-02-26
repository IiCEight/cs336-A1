import torch
import torch.nn as nn

from cs336_basics.module.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.module.rms_norm import RMSNorm
from cs336_basics.module.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, max_seq_len:int=None, 
                 theta :float = None):
        """
        Args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln_1 = RMSNorm(d_model)
        self.ln_2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor)->torch.Tensor:
        y = x + self.attn(self.ln_1(x), token_positions)
        return y + self.ffn(self.ln_2(y))