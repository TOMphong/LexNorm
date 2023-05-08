#from Data import *
#from Engine import *
#from Model import *
#from Utils import *
#from Data import *
#from torch.nn import functional as F

from Builders.build_trainer import build_trainer

#from torch.utils.data import DataLoader  ## -> builder.py ???
from Configs.BaseConfig import BaseConfig

if __name__ == "__main__":
    config = BaseConfig()

    trainer, pretrain = build_trainer(config)
    
    trainer.train(pretrain=pretrain)

    