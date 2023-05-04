import math 
import torch
from torch import nn
from torch.nn import (TransformerEncoderLayer, 
                      TransformerDecoderLayer, 
                      TransformerEncoder, 
                      TransformerDecoder, 
                      Embedding)
from Data.vocab import Vocab
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class Encoder(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 nhead = 8, 
                 dropout = 0.1, 
                 batch_first=True, 
                 num_encoder = 1,
                 src_vocab = 3000,
                 max_src_len = 5000 
                 ):
        super(Encoder, self).__init__()
        self.e_embedding = Embedding(src_vocab, d_model)
        self.e_pos_embed = PositionalEncoding(d_model, dropout, max_src_len)

        
        #encoder-decoder
        
        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model = d_model, nhead=nhead, 
                                                      dropout=dropout, batch_first=True) 
                                          , num_encoder)

    def forward(self, src):
        e_embed = self.e_pos_embed(self.e_embedding(src))

        enc_out = self.encoder(e_embed)

        return enc_out
      


class Decoder(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 nhead = 8, 
                 dropout = 0.1, 
                 batch_first=True, 
                 dim_feedforward = 2048,
                 num_decoder = 1,
                 tgt_vocab = 5000,
                 max_tgt_len = 5000
                 ):
        super(Decoder, self).__init__()

        self.d_embedding = Embedding(tgt_vocab, d_model)
        self.d_pos_embed = PositionalEncoding(d_model, dropout, max_tgt_len)

        self.decoder = TransformerDecoder(TransformerDecoderLayer(d_model = d_model, nhead=nhead, 
                                                     dropout=dropout, batch_first=True, 
                                                     dim_feedforward = dim_feedforward),
                                          num_decoder)
        #fc_out
        self.fc_out = nn.Linear(d_model, tgt_vocab)
    def forward(self, tgt ,enc_out, mask=None):
        d_embed = self.d_pos_embed(self.d_embedding(tgt))

        out = self.decoder(d_embed, enc_out, mask)
        
        out = F.softmax(self.fc_out(out), dim=-1)
        
        return out

class Transformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder:int = 1,
                 num_decoder:int = 1,
                 dropout:int = 0.1,
                 max_src_len:int = 5000,
                 max_tgt_len:int = 5000,
                 dim_feedforward:int = 2048,
                 src_vocab:int = 3000,
                 tgt_vocab:int = 3000,
                 batch_first:bool = True
                 ):
        super(Transformer, self).__init__()
        
        self.encoding = Encoder(d_model, nhead, dropout, batch_first, num_encoder, src_vocab, max_src_len)
        
        self.decoding = Decoder(d_model, nhead, dropout, batch_first, dim_feedforward, num_decoder, tgt_vocab, max_tgt_len)


    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones((tgt_len, tgt_len))).expand(tgt_len, tgt_len)
        return tgt_mask

    def forward(self, src, tgt, device='cuda'):
        tgt_mask = self.make_tgt_mask(tgt).to(device)

        enc_out = self.encoding(src)
        
        outputs = self.decoding(tgt, enc_out, tgt_mask)
    
        return outputs
        

    def decode(self, src, max_len_out = 50): # with torch.no_grad()

        tgt_mask = self.make_tgt_mask(src)

        enc_out = self.encoding(src)

        out_labels = [Vocab.BOS]
        batch_size, seq_len = src.shape[0], src.shape[1]
       
        iter = 1

        out = torch.Tensor([[Vocab.BOS]])
        out=out.to(torch.long)
  
        while True:
            out = F.pad(torch.Tensor(out), (0, seq_len - len(out)),'constant')

            if iter >= max_len_out-1:
                out_labels.append(Vocab.EOS)
                break            
            out = self.decoding(out, enc_out, tgt_mask)
            
            out = out[:,-1,:]
            
            out = out.argmax(-1)
            
            outiter = out.item()
            out_labels.append(outiter)

            if outiter == Vocab.EOS:
                break

            out = torch.unsqueeze(out, axis=0)
            
            iter+=1
        return out_labels
