
import torch


def softmax(x:torch.Tensor, dim:int=-1)->torch.Tensor:
    """
        dim (int): Dimension of the x to apply softmax to.
    """
    shift_x = x - x.max(dim = dim, keepdim=True).values
    exp_x = torch.exp(shift_x)
    denominator = torch.reciprocal(torch.sum(exp_x, dim = dim, keepdim= True))
    return denominator * exp_x
