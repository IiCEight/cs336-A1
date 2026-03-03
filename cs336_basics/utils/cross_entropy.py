

from einops import rearrange, repeat
from loguru import logger
import torch


def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
    """
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    """
    logger.debug("Cross entropy loss. Inputs shape: {}, targets shape: {}", inputs.shape, targets.shape)

    max_inputs = inputs.max(dim=-1, keepdim=True).values
    
    # PyTorch automatically expands max_input to match input's shape here
    shift_inputs = inputs - max_inputs
    
    # 2. Compute LogSumExp (keepdim=True so we can add max_input back!)
    log_sum_exp = shift_inputs.exp().sum(dim=-1, keepdim=True).log() + max_inputs

    # 3. Safely extract the correct logits using gather
    # targets_unsqueeze = targets.unsqueeze(-1)
    targets_extend = repeat(targets, "... -> ... 1")
    # Use gather to index,correct_logits[i] = inputs[i, targets_extend[i,0]]
    correct_logits = torch.gather(inputs, dim=-1, index=targets_extend)
    
    # 4. Compute the final loss
    loss = -correct_logits + log_sum_exp
    
    # 5. Return the average as required!
    return loss.mean()
