from .build_model import build_model
from .build_data import build_data
from Configs.BaseConfig import BaseConfig
from Utils import my_collate
from Engine import Trainer

from torch.utils.data import DataLoader
import torch
from torch import nn



config = BaseConfig()

def build_trainer(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = build_model(config)
    traindata = DataLoader(dataset = data, 
                           batch_size = config.ENGINE.TRAINER.batch_size, 
                           collate_fn = my_collate, 
                           shuffle = config.ENGINE.TRAINER.shuffle)
    criterion =  nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(params = model.parameters(), 
                                 betas = config.ENGINE.TRAINER.betas, 
                                 eps = config.ENGINE.TRAINER.eps, 
                                 lr = config.ENGINE.TRAINER.lr)

    trainer = Trainer(model = model,
                     criterion = criterion,
                     optim = optimizer,
                     epochs = config.ENGINE.TRAINER.epochs,
                     dataloader = traindata,
                     device = device)
    
    pretrain = config.ENGINE.TRAINER.pretrain

    return trainer, pretrain
