

import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas = (0.9, 0.999), eps= 1e-8, weight_decay=0.01):

        default = dict(lr = lr, betas = betas, eps = eps, lam = weight_decay)

        # PyTorch takes your raw params and your defaults dictionary 
        # and fuses them into a structured list called self.param_groups.
        super().__init__(params, default)

    @torch.no_grad() # CRITICAL: We don't want PyTorch to track these updates!
    def step(self, closure = None):
        # The API specifies that the user might pass in a callable closure to 
        # re-compute the loss before the optimizer step. We won't need this for the 
        # optimizers we’ll use, but we add it to comply with the API.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Loops through your layer groups. 
        # If you passed all model.parameters() at once, there is only one group
        for group in self.param_groups:
            # This 
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["lam"]

            # Iterate over the specific parameters (weights/biases) in this group
            # e.g., conv1.weight, bn1.weight, bn1.bias and so on.
            for p in group['params']:
                # If this parameter doesn't have a gradient, skip it
                if p.grad is None:
                    continue
                # Get state associated with p.
                # Used to store and initialize these moment vectors.
                state = self.state[p]
                # 2. Lazy Initialization (Only runs at t=0!)
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                
                # Extract state variables
                t = state["t"]
                m = state["m"]
                v = state["v"]

                # Get the actual gradient data
                g = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                alpha_t = alpha * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= alpha * lam * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
            
        return loss

