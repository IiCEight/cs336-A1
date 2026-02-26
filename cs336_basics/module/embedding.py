import math
from loguru import logger
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim:int, 
                 device:torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        """
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors
            ...
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # It is a map from token_id (the number of column) 
        # to embedding vector (that column vector of embedding_matrix)
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        self.reset_parameter()

    def reset_parameter(self):

        std = math.sqrt(2.0/(self.num_embeddings + self.embedding_dim))

        nn.init.trunc_normal_(self.embedding_matrix, 0, std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # Look up from token_id to embedding vector

        logger.debug("shape of result {}", self.embedding_matrix[token_ids].shape)
        return self.embedding_matrix[token_ids]

