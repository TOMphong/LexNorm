import torch.nn.functional as F
from torch import nn

class BaseConfig():
    #TRANSFORMER
    def __init__(self):
        #==============DATA=================
        
        ##VOCAB 
        self.DATA_VOCAB_name = "default"

        ##DATASET
        self.DATA_DATASET_train = ""
        #self.DATA.DATASET.vad = ""
        #self.DATA.DATASET.test = ""
        self.DATA_DATASET_truncate_src = False
        self.DATA_DATASET_max_src_len = None

        self.DATA_DATASET_truncate_tgt = False
        self.DATA_DATASET_max_tgt_len = None
        self.DATA_DATASET_max_rows = 1000
        self.DATA_DATASET_min_preq = 0

        #==============ENGINE=================
        
        #Dataloader
        self.ENGINE_TRAINER_batch_size = 100
        self.ENGINE_TRAINER_shuffle = True
        
        #Loss
                
        #Optim
        ###Adam
        self.ENGINE_TRAINER_betas = (0.9, 0.98)
        self.ENGINE_TRAINER_eps = 1e-09
        self.ENGINE_TRAINER_lr = 0.01

        # Trainer
        self.ENGINE_TRAINER_checkpoint = "Model/models/model.pth"  # to save params
        self.ENGINE_TRAINER_epochs = 5
        self.ENGINE_TRAINER_pretrain = ""                        # to load params

        #==============MODEL=================        

        #self.MODEL.input_vocab_size   #depends on the data
        #self.MODEL.output_vocab_size  #depends on the data
        #self.MODEL.max_positions = 512
        self.MODEL_num_e_blocks = 1
        self.MODEL_num_d_blocks = 1
        self.MODEL_num_heads = 8
        self.MODEL_d_model = 512
        self.MODEL_dim_pffn = 2048
        self.MODEL_dropout = 0.1

        