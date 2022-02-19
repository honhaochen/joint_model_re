import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *
import numpy as np


class Unlabelled_Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset1, dataset2):
        
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.text_list1 = []
        self.text_list2 = []

        for line in open(os.path.join(self.data_root, dataset1), 'r'):
            line = line.strip("\n")
            instance = line.split() # default split by space
            self.text_list1.append(instance)
        
        for line in open(os.path.join(self.data_root, dataset2), 'r'):
            line = line.strip("\n")
            instance = line.split() # default split by space
            self.text_list2.append(instance)

    def __getitem__(self, index):
        dataset1 = self.text_list1[index]
        dataset2 = self.text_list2[index]
        tokens1 = self.text2tensor(dataset1)
        tokens2 = self.text2tensor(dataset2)
        return dataset1, tokens1, dataset2, tokens2
    
    def __len__(self):
        assert len(self.text_list1) == len(self.text_list2)
        return len(self.text_list1)

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        padded_list = list(map(lambda x: 1, text))
        padded_list.extend([0] * (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.dataset1 = transposed_data[0]
        self.tokens1 = pad_sequence(transposed_data[1], batch_first=True)
        self.dataset2 = transposed_data[2]
        self.tokens2 = pad_sequence(transposed_data[3], batch_first=True)

def collate_fn(batch):
    return Batch_reader(batch)


Unlabelled_Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=False)
