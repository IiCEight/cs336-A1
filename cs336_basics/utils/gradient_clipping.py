

from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, 
                      eps :float = 1e-6)->None:
    """ The gradients of the parameters (parameter.grad) will be modified in-place."""

    # 1. Safely collect all valid gradients into a concrete list
    # NOTE: `parameters` may acts as a Python generator,
    # When you try to run your second loop `for param in parameters:`, 
    # the generator is empty, and the loop will silently do nothing!
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    
    # NOTE: We calculate the gradient for all parameters
    # 2. Flatten all gradients to 1D before concatenating to avoid shape crashes!
    flattened_grads = torch.concat([g.view(-1) for g in grads])
    
    # 3. Calculate the global norm using p=2 (standard L2 norm)
    norm_l2 = torch.norm(flattened_grads, p=2)

    print("norm = ", norm_l2)

    if norm_l2 > max_l2_norm:
        scale = max_l2_norm / (norm_l2 + eps)

        # 5. FIX: Iterate over the 'grads' list, not the 'parameters' iterable
        for g in grads:
            g.mul_(scale)