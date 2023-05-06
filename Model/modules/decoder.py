import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dim_pwff:  int,
                 dropout: float) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        # Target-source
        self.tgt_src_attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Position-wise
        self.feed_forward = PositionWiseFeedForward(d_model, dim_pwff, dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, y, y_mask, x, x_mask) -> Tensor:
        y = y + self.sub_layer1(y, y_mask)
        y = y + self.sub_layer2(y, x, x_mask)
        y = y + self.sub_layer3(y)
        return y

    def sub_layer1(self, y: Tensor, y_mask: Tensor) -> Tensor:
        y = self.layer_norm1(y)
        y = self.self_attn(y, y, y_mask)
        return y

    def sub_layer2(self, y: Tensor, x: Tensor, x_mask: Tensor) -> Tensor:
        y = self.layer_norm2(y)
        y = self.tgt_src_attn(y, x, x_mask)
        return y

    def sub_layer3(self, y: Tensor) -> Tensor:
        y = self.layer_norm3(y)
        y = self.feed_forward(y)
        return y


class Decoder(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 num_heads:  int,
                 d_model:  int,
                 dim_pffn:   int,
                 dropout:  float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [DecoderBlock(num_heads, d_model, dim_pffn, dropout)
             for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        for block in self.blocks:
            y = block(y, y_mask, x, x_mask)
        y = self.layer_norm(y)
        return y