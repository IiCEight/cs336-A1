from einops import einsum
import math
from loguru import logger
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features:int, out_features: int, device:torch.device = None, 
                 dtype:torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        #  We use nn.Parameter so PyTorch knows to track gradients for these
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameter()

    def reset_parameter(self):

        std = math.sqrt(2.0/(self.in_features + self.out_features))

        nn.init.trunc_normal_(self.weight, 0, std, a = -3 * std, b = 3 * std)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # NOTE: row vector is one sample in x.

        logger.debug("Shape of x {}", x.shape)

        y = einsum(x, self.weight, "... in_features, out_features in_features -> " \
            "... out_features")
        
        return y
