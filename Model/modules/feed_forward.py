import torch.nn as nn
from torch import Tensor

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_embed: int, dim_pffn: int, dropout: float) -> None:
        super(PositionWiseFeedForward, self).__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_pffn, dim_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pffn(x)