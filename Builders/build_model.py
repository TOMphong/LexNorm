from Configs.BaseConfig import BaseConfig
from .build_data import build_data
from Model.BaseTransformer import BaseTransformer


config = BaseConfig()

def build_model(config):
    data = build_data(config)
    src_vocab = len(data.vocab)
    tgt_vocab = len(data.vocab)
    model = BaseTransformer(input_vocab_size = src_vocab,
                            output_vocab_size = tgt_vocab,
                            max_positions = data.src_len+2,
                            num_e_blocks = config.MODEL.num_e_blocks,
                            num_e_blocks = config.MODEL.num_d_blocks,
                            num_heads = config.MODEL.num_heads,
                            d_model = config.MODEL.d_model,
                            dim_pffn = config.MODEL.dim_pffn,
                            dropout = config.MODEL.dropout)
    
    return model, data