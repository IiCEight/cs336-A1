
from einops import einsum
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        
        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        """
        # NOTE: We need to upcase the dtype to torch.float32 to prevent overflowing

        original_dtype = x.dtype
        x = x.to(torch.float32)

        # 1. Calculate the Mean of Squares (RMS part)
        # x.pow(2) -> a_i^2
        # mean(-1) -> Sum / d_model
        # keepdim=True -> Ensures we can broadcast back to the original shape
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 2. Calculate the Inverse Root Mean Square
        # rsqrt(y) is equivalent to 1 / sqrt(y)
        inv_RMS = torch.rsqrt(mean_square + self.eps)
        
        # 3. Normalize and Scale
        # (x * inv_rms) is the "RMSNorm(a_i)" part
        # * self.weight is the "g_i" part

        # return einsum(x * inv_RMS,self.weight, "... d_model, d_model -> ... d_model")
        y = (x * inv_RMS) * self.weight
        return y.to(original_dtype)



