import math
import torch
import torch.nn as nn
from torch import Tensor

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(Embedding, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)        
        self.sqrt_d = math.sqrt(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x.long())    
        x = x * self.sqrt_d
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, max_positions: int, d_model: int, dropout: float) -> None:
        super().__init__()

        assert d_model % 2 == 0

        # Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        position = torch.arange(max_positions).unsqueeze(1)
        dim_pair = torch.arange(0, d_model, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension: (1, max_positions, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as non-learnable parameters
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        # Max sequence length within the current batch
        max_sequence_length = x.size(1)
        
        # Add positional encoding up to the max sequence length
        x = x + self.pe[:, :max_sequence_length]
        x = self.dropout(x)
        return x
