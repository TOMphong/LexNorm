import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward

class EncoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dim_pwff:  int,
                 dropout: float) -> None:
        super(EncoderBlock, self).__init__()

        # Self-attention
        self.self_atten = MultiHeadAttention(num_heads, d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        # Point-wise feed-forward
        self.feed_forward = PositionWiseFeedForward(d_model, dim_pwff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = x + self.sub_layer1(x, x_mask)
        x = x + self.sub_layer2(x)
        return x

    def sub_layer1(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = self.layer_norm1(x)
        x = self.self_atten(x, x, x_mask)
        return x
    
    def sub_layer2(self, x: Tensor) -> Tensor:
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 num_heads:  int,
                 d_model:  int,
                 dim_pffn:   int,
                 dropout:  float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [EncoderBlock(num_heads, d_model, dim_pffn, dropout)
             for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: Tensor, x_mask: Tensor):
        for block in self.blocks:
            x = block(x, x_mask)
        x = self.layer_norm(x)
        return x