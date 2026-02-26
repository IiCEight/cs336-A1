import torch
import torch.nn as nn

# NOTE: this layer has no learnable parameters.
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta: float \theta value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # pre_compute sin and cos

        # 1. Calculate the denominator: Theta ^ ((2k - 2) / d)
        # In Python (0-indexed), torch.arange(0, d, 2) gives us 0, 2, 4... d-2.
        # This perfectly matches the (2k - 2) from the math formula where k starts at 1.
        exponent = torch.arange(0, d_k, 2).float() / d_k
        inv_freq = 1.0 / (theta ** exponent)
        
        # 2. Get the token positions 'i' (0, 1, 2, ..., max_seq_len - 1)
        positions = torch.arange(max_seq_len).float()
        
        # 3. Calculate theta_{i, k} by multiplying positions (i) and inv_freq
        # torch.outer computes the outer product, resulting in a 2D matrix 
        # of shape (max_seq_len, d/2). Awesome step!!!
        angles = torch.outer(positions, inv_freq)
        
        # 4. To apply this easily to our queries/keys, we duplicate the angles 
        # so the shape matches the full dimension 'd' (max_seq_len, d)
        # This repeats each angle side-by-side: [a, b ...] -> [a, a, b, b ...]
        angles = torch.repeat_interleave(angles, 2, dim=-1)

        # 5. Pre-compute the sine and cosine values
        cos_cached = angles.cos()
        sin_cached = angles.sin()
        
        # 6. Register as non-persistent buffers
        # This ensures they are moved to the correct device (GPU/CPU) alongside 
        # the model, but are NOT saved in the state_dict checkpoint.
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def rotate_adjacent(self, x: torch.Tensor) -> torch.Tensor:
            """Rotates adjacent pairs of the last dims: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]"""
            # Get even and odd indices
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            
            # Stack them as [-odd, even] and flatten the last two dimensions back out
            return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        # Slice the cached buffers up to the current sequence length
        # NOTE:
        # the resulting cos and sin tensors will AUTOMATICALLY be shape (..., seq_len, d_k).
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # Apply the RoPE math to the x
        # This is specially optimized. And it is equivalent to math in page 23
        x_rotated = (x * cos) + (self.rotate_adjacent(x) * sin)
        
        return x_rotated
