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
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_positions: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
