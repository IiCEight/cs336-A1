import math

from einops import rearrange
from einops import einsum
import torch
import torch.nn as nn

from cs336_basics.module.rope import RoPE
from cs336_basics.utils.scaled_dot_product_attention import scaled_dot_product_attention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len:int=None, 
                 theta :float = None, device:torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        # we concate qkv into a whole matrix since d_k == d_v
        self.weight_qkv = nn.Parameter(torch.empty((num_heads * self.d_k * 3, d_model)))
        self.weight_O = nn.Parameter(torch.empty(d_model, num_heads * self.d_v))

        if theta is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

        self.reset_parameter()

    def reset_parameter(self):
        # note: This depends on d_k == d_v
        std = math.sqrt(2.0/(self.d_k * self.num_heads + self.d_model))

        nn.init.trunc_normal_(self.weight_qkv, 0, std, a = -3 * std, b = 3 * std)
        nn.init.trunc_normal_(self.weight_O, 0, std, a = -3 * std, b = 3 * std)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor = None)->torch.Tensor:
        input = einsum(x, self.weight_qkv, "... seq_len d_model, d_out d_model -> ... seq_len d_out")
        # Regard qkv and num_heads as batch-like dimensions.
        input = rearrange(
            input, "... seq_len (qkv num_heads d_k) -> ... qkv num_heads seq_len d_k",
            qkv = 3,
            num_heads = self.num_heads,
            d_k = self.d_k
            )
        # reshape token_positions to be broadcastable with input.
        # add one dim for num_heads.
        if token_positions is not None:
            token_positions = rearrange(token_positions, "... seq_len-> ... 1 seq_len")
        
        seq_len = input.shape[-2]
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).to(x.device)
        causal_mask = ~causal_mask.triu(diagonal=1)
        q, k, v = input[...,0,:,:,:], input[...,1,:,:,:], input[...,2,:,:,:]
        if token_positions is not None:
            q = self.rope.forward(q, token_positions)
            k = self.rope.forward(k, token_positions)

        attention_x = scaled_dot_product_attention(q, k, v, causal_mask)
        # Concatenate all heads.
        attention_x = rearrange(
            attention_x, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
            num_heads = self.num_heads,
            seq_len =seq_len,
            d_v =self.d_v
            )

        return einsum(attention_x,self.weight_O, "... hd_v, d_model hd_v -> ... d_model")



