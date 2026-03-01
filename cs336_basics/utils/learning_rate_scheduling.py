


import math


def lr_cosine_schedule(t, max_learning_rate, 
                       min_learning_rate, warmup_iters, cosine_cycle_iters):
    """
    NOTE: t start from 1
    """
    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate
    elif t <= cosine_cycle_iters:
        return (
            min_learning_rate + 
            (1.0 + math.cos((t - warmup_iters) / 
                            (cosine_cycle_iters - warmup_iters) * math.pi)) *
                (max_learning_rate - min_learning_rate) / 2
        )
    else:
        return min_learning_rate
