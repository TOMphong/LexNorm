import torch.nn as nn
from torch import Tensor
from modules import Embedding, PositionalEncoding, Encoder, Decoder

class BaseTransformer(nn.Module):
    def __init__(self,
                 input_vocab_size:  int,
                 output_vocab_size: int,
                 max_positions:     int,
                 num_e_blocks:        int,
                 num_d_blocks:        int,
                 num_heads:         int,
                 d_model:         int,
                 dim_pffn:          int,
                 dropout:         float) -> None:
        super(BaseTransformer, self).__init__()

        # Input embeddings, positional encoding, and encoder
        self.input_embedding = Embedding(input_vocab_size, d_model)
        self.input_pos_encoding = PositionalEncoding(
                                      max_positions, d_model, dropout)
        self.encoder = Encoder(num_e_blocks, num_heads, d_model, dim_pffn, dropout)

        # Output embeddings, positional encoding, decoder, and projection 
        # to vocab size dimension
        self.output_embedding = Embedding(output_vocab_size, d_model)
        self.output_pos_encoding = PositionalEncoding(
                                       max_positions, d_model, dropout)
        self.decoder = Decoder(num_d_blocks, num_heads, d_model, dim_pffn, dropout)
        self.projection = nn.Linear(d_model, output_vocab_size)

        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: Tensor, y: Tensor,
                      x_mask: Tensor=None, y_mask: Tensor=None) -> Tensor:
        x = self.encode(x, x_mask)
        y = self.decode(x, y, x_mask, y_mask)
        return y

    def encode(self, x: Tensor, x_mask: Tensor=None) -> Tensor:
        x = self.input_embedding(x)
        x = self.input_pos_encoding(x)
        x = self.encoder(x, x_mask)
        return x

    def decode(self, x: Tensor, y: Tensor,
                     x_mask: Tensor=None, y_mask: Tensor=None) -> Tensor:
        y = self.output_embedding(y)
        y = self.output_pos_encoding(y)
        y = self.decoder(x, x_mask, y, y_mask)
        return self.projection(y)