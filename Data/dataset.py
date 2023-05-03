import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from Vocab.vocab import Vocab
from collections import Counter

class MyDataset(Dataset):
  def __init__(self, filename:str, tokenize = None, max_src_len: int=None, 
               max_tgt_len: int=None, max_rows: int=None, truncate_src: bool=False, truncate_tgt: bool=False):
      super(MyDataset, self).__init__()
      print("Reading dataset {}...".format(filename), end=' ', flush=True)
      
      assert tokenize!=None, "Need a function for tokenizing"
      
      self.filename = filename
      self.pairs = []
      self.src_len = 0
      self.tgt_len = 0
      self.max_rows = max_rows 


      if max_rows is None:
        df = pd.read_csv(filename, encoding ='utf-8')
      else:
        df = pd.read_csv(filename, encoding='utf-8', nrows=max_rows)
      
      sources = df['history_text'].apply(lambda x: tokenize(x))

      if truncate_src:
        sources = [src[:max_src_len] if len(src)>max_src_len else src for src in sources]
      
      targets = df['real_name'].apply(lambda x: tokenize(x))

      if truncate_tgt:
        targets = [tgt[:max_tgt_len] if len(tgt)>max_tgt_len else tgt for tgt in targets]

      src_length = [len(src) for src in sources]
      tgt_length = [len(tgt) for tgt in targets]

      max_src = max(src_length)
      max_tgt = max(tgt_length)

      self.src_len = max_src
      self.tgt_len = max_tgt

      self.pairs.append([(src, tgt, src_len, tgt_len) for src, tgt, src_len, tgt_len in zip(sources, targets, src_length, tgt_length)])
      self.pairs = self.pairs[0]
      print(f"{len(self.pairs)} pairs were built.", flush=True)

      self.build_vocab()

  def build_vocab(self, name: str = "vocab", min_freq: int = 0, save_vocab: bool=False, save_dir:str = "vocab.pth")->Vocab:
      total_words = [src+tgt for src,tgt, len_src, len_tgt in self.pairs]
      total_words = [item for sublist in total_words for item in sublist]
      word_counts = Counter(total_words)
      vocab = Vocab(name=name)
      for word, count in word_counts.items():
        if(count > min_freq):
          vocab.add_by_words([word])

      print(f"Vocab {name} of dataset {self.filename} was created.", flush=True)

      if save_vocab:
        vocab.save_to_file(save_dir)
        print(f"Saved vocab {name} to {save_dir}.", flush=True)

      self.vocab=vocab
      return vocab
        
  def vectorize(self, tokens):
      return [self.vocab[token] for token in tokens]
      
  def unvectorize(self, indices):
      return [self.vocab[i] for i in indices]
  def __getitem__(self, index):
      return { 'x':self.vectorize(self.pairs[index][0]),
              'y':self.vectorize(self.pairs[index][1]),
              'src':self.pairs[index][0],
              'tgt':self.pairs[index][1],
              'x_len':len(self.pairs[index][0]),
              'y_len':len(self.pairs[index][1])}
  def __len__(self):
      return len(self.pairs)