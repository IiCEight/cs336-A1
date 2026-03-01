import math

from einops import einsum
import torch
import torch.nn as nn


# See page 22 for details.
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device:torch.device | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.weight_2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.weight_1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.weight_3 = nn.Parameter(torch.empty(d_ff, d_model))

        self.reset_parameter()

    def reset_parameter(self):
        std = math.sqrt(2.0/(self.d_model + self.d_ff))

        nn.init.trunc_normal_(self.weight_1, 0, std, a = -3 * std, b = 3 * std)
        nn.init.trunc_normal_(self.weight_2, 0, std, a = -3 * std, b = 3 * std)
        nn.init.trunc_normal_(self.weight_3, 0, std, a = -3 * std, b = 3 * std)

    def SiLU(self, x:torch.Tensor)->torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        W1_x = einsum(x, self.weight_1, "... d_model, d_ff d_model -> ... d_ff")
        # x = self.SiLU(x)
        # In PyTorch, torch.sigmoid(x) is an element-wise operation.
        # So it's ok if even if x is batched.
        SiLU_x = W1_x * torch.sigmoid(W1_x)
        W3_x = einsum(x, self.weight_3, "... d_model, d_ff d_model -> ... d_ff")
        
        x = einsum(SiLU_x, W3_x, "... d_ff, ... d_ff -> ... d_ff")

        return einsum(x, self.weight_2, "... d_ff, d_model d_ff -> ... d_model")

        


