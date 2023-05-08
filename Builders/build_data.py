from Configs.BaseConfig import BaseConfig
from Utils import tokenizing
from Data import *

config = BaseConfig()

def build_data(config):
    vocab = Vocab(config.DATA_VOCAB_name)
    data = MyDataset(filename = config.DATA_DATASET_train,
                    tokenize = tokenizing,
                    max_rows = config.DATA_DATASET_max_rows,
                    truncate_src = config.DATA_DATASET_truncate_src,
                    max_src_len = config.DATA_DATASET_max_src_len,
                    truncate_tgt = config.DATA_DATASET_truncate_tgt,
                    max_tgt_len = config.DATA_DATASET_max_tgt_len,
                    min_freq = config.DATA_DATASET_min_freq)
    
    return data

