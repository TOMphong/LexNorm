import torch.nn.functional as F
from torch import nn

class BaseConfig():
    #TRANSFORMER
    def __init__(self):
        #==============DATA=================
        
        ##VOCAB 
        self.DATA.VOCAB.name = "default"

        ##DATASET
        self.DATA.DATASET.train = ""
        #self.DATA.DATASET.vad = ""
        #self.DATA.DATASET.test = ""
        self.DATA.DATASET.truncate_src = False
        self.DATA.DATASET.max_src_len = None

        self.DATA.DATASET.truncate_tgt = False
        self.DATA.DATASET.max_tgt_len = None
        self.DATA.DATASET.max_rows = 1000
        self.DATA.DATASET.min_preq = 0

        #==============ENGINE=================
        
        #Dataloader
        self.ENGINE.TRAINER.batch_size = 100
        self.ENGINE.TRAINER.shuffle = True
        
        #Loss
                
        #Optim
        ###Adam
        self.ENGINE.TRAINER.betas = (0.9, 0.98)
        self.ENGINE.TRAINER.eps = 1e-09
        self.ENGINE.TRAINER.lr = 0.01

        # Trainer
        self.ENGINE.TRAINER.checkpoint = "Model/models/model.pth"  # to save params
        self.ENGINE.TRAINER.epochs = 5
        self.ENGINE.TRAINER.pretrain = ""                        # to load params

        #==============MODEL=================        

        #self.MODEL.input_vocab_size   #depends on the data
        #self.MODEL.output_vocab_size  #depends on the data
        #self.MODEL.max_positions = 512
        self.MODEL.num_e_blocks = 1
        self.MODEL.num_d_blocks = 1
        self.MODEL.num_heads = 8
        self.MODEL.d_model = 512
        self.MODEL.dim_pffn = 2048
        self.MODEL.dropout = 0.1

        