import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    sqrt_d = query.shape[-1]**0.5

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_d
    
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    
    weight = F.softmax(scores, dim=-1)    
    return torch.matmul(weight, value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.dim_embed = d_model
        self.dim_head = d_model // num_heads

        self.query  = nn.Linear(d_model, d_model)
        self.key    = nn.Linear(d_model, d_model)
        self.value  = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor=None) -> Tensor:
        query = self.query(x)
        key   = self.key  (y)
        value = self.value(y)

        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_head)
        key   = key  .view(batch_size, -1, self.num_heads, self.dim_head)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head)

        
        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

        attn = attention(query, key, value, mask)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embed)
        
        out = self.dropout(self.output(attn))

        return out