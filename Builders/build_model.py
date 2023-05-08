from .build_data import build_data
from Model.BaseTransformer import BaseTransformer



def build_model(config):
    data = build_data(config)
    src_vocab = len(data.vocab)
    tgt_vocab = len(data.vocab)
    model = BaseTransformer(input_vocab_size = src_vocab,
                            output_vocab_size = tgt_vocab,
                            max_positions = data.src_len+2,
                            num_e_blocks = config.MODEL_num_e_blocks,
                            num_d_blocks = config.MODEL_num_d_blocks,
                            num_heads = config.MODEL_num_heads,
                            d_model = config.MODEL_d_model,
                            dim_pffn = config.MODEL_dim_pffn,
                            dropout = config.MODEL_dropout)
    
    return model, data