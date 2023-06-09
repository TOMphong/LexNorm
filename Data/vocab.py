from collections import Counter
import pickle
import re

# assume the input is preprocessed
class Vocab(object):

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    SEP = 4
    
    def __init__(self, 
                 name:str = "default"):
        self.name = name
        self.word2index = {"<PAD>":0, "<BOS>":1, "<EOS>":2, "<UNK>":3, "<SEP>":4}
        self.word2count = Counter()
        self.index2word = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"]
        self.numword = 5
        self.numsentence = 0

        print(f"\n ## Vocabulary {self.name} is created.\n", flush=True)

    def add_by_words(self, 
                     words: list):  # list of words
        for word in words:
          if word not in self.index2word:
            self.index2word.append(word)
            self.word2index[word] = self.numword
            self.numword += 1
        self.word2count.update(words)

    def __getitem__(self, item):
      try:
        if type(item) is int:
          if item >= self.numword:
            return self.index2word[self.UNK]
          
          return self.index2word[item]  
        return self.word2index[item]
      
      except:
        return "<UNK>"
    def __len__(self):
        return self.numword
    
    def save_to_file(self, filename):
        with open(filename,'wb') as f:
            pickle.dump(self,f) 
    
    def load_vocab(filename):
        with open(filename,'rb') as f:
            v = pickle.load(f)
        return v
