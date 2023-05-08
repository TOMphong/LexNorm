from Configs.BaseConfig import BaseConfig
from Utils import tokenizing
from Data import *

config = BaseConfig()

def build_data(config):
    vocab = Vocab(config.DATA.VOCAB.name)
    data = MyDataset(filename = config.DATA.DATASET.train,
                    tokenize = tokenizing,
                    max_rows = config.DATA.DATASET.max_rows,
                    truncate_src = config.DATA.DATASET.truncate_src,
                    max_src_len = config.DATA.DATASET.max_src_len,
                    truncate_tgt = config.DATA.DATASET.truncate_tgt,
                    max_tgt_len = config.DATA.DATASET.max_tgt_len,
                    min_freq = config.DATA.DATASET.min_freq)
    
    return data

