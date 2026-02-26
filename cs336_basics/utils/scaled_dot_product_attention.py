from einops import einsum
import torch

from cs336_basics.utils.softmax import softmax


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    d_k = Q.shape[-1]
    numerator = einsum(Q, K, "... query_len d_k, ... key_len d_k -> ... query_len key_len")
    Q_transpoesK_scaled = numerator * torch.rsqrt(torch.tensor(d_k))
    
    if mask is not None:
        Q_transpoesK_scaled = Q_transpoesK_scaled.masked_fill(mask == False, float("-inf"))

    tmp = softmax(Q_transpoesK_scaled, dim = -1)

    return einsum(tmp, V, "... query_len key_len, ... key_len d_v -> ... query_len d_v")

    
