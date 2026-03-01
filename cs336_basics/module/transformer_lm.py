import torch
import torch.nn as nn

from cs336_basics.module.embedding import Embedding
from cs336_basics.module.linear import Linear
from cs336_basics.module.rms_norm import RMSNorm
from cs336_basics.module.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, 
                 d_model:int, num_heads:int, d_ff: int, rope_theta : float, 
                 device:torch.device | None = None):
        """
        Args:
            vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
                embedding matrix.
            context_length: int The maximum context length, necessary for determining the dimensionality of
                the position embedding matrix.
            num_layers: int The number of Transformer blocks to use.
            d_model: int The dimensionality of the model embeddings and sublayer outputs
            num_heads: int: Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff : int Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta : float The RoPE $\Theta$ parameter.
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta  = rope_theta
        self.embedding = Embedding(vocab_size, d_model, device)
        # NOTE: Wrapped in nn.ModuleList so PyTorch tracks the weights!
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device) 
            for _ in range(num_layers)
        ])
        self.norm_layer = RMSNorm(d_model, device)
        self.linear = Linear(d_model, vocab_size, device)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor):
        x = self.embedding.forward(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, token_positions)        
        x = self.norm_layer(x)
        x = self.linear(x)
        return x

