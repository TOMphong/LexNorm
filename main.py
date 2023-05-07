from Data import *
from Engine import *
from Model import *
from Utils import *
from Data import *
from torch.nn import functional as F


from torch.utils.data import DataLoader  ## -> builder.py ???

class Params():
    #TRANSFORMER
    D_MODEL = 512

    EMBED = 300
    HEAD = 8

    ENCODER_LAYER = 1
    DECODER_LAYER = 1
    TGT_VOCAB = 5000

    #TRAINER
    LR = 0.01
    NUM_EPOCHS = 20
    BATCH_SIZE = 50

    #OPTIM _ ADAM
    BETA =  (0.9, 0.98)
    EPS = 1e-09

    #DATA
    TRAIN_PATH = ''
    VAL_PATH = ''
    TEST_PATH = ''
      #MAX_SQ_LEN, TRUNCATE, ...
    
    MODEL_PATH = ''

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = Params()
    data = MyDataset( '/content/heroes.csv' ,tokenizing, max_rows=10, truncate_src=True, max_src_len=510)

    traindata = DataLoader(data, batch_size = 2, collate_fn = my_collate, shuffle=True)

    src_vocab = len(data.vocab)
    tgt_vocab = len(data.vocab)

    #model = Transformer(src_vocab = src_vocab, tgt_vocab = tgt_vocab)
    model = BaseTransformer(input_vocab_size=src_vocab,
                 output_vocab_size=tgt_vocab,
                 max_positions = 3000,
                 num_e_blocks=1,
                 num_d_blocks=1,
                 num_heads=8,
                 d_model=512,
                 dim_pffn=2048,
                 dropout=0.1)
    criterion =  nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09, lr=0.01)

    t = Trainer(model, criterion, optimizer, 3, traindata, device)

    t()