import json
import os

from dataclasses import dataclass

@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.model: str
        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.test: str
        self.relation_vocab: str
        self.model_dir: str
        self.print_epoch: int
        self.evaluation_epoch: int
        self.max_text_len: int
        self.cell_name: str
        self.emb_size: int
        self.char_emb_size: int
        self.rel_emb_size: int
        self.hidden_size: int
        self.char_encoder_hidden_size: int
        self.threshold: float
        self.activation: str
        self.optimizer: str
        self.learning_rate: int
        self.epoch_num: int
        self.gpu: int
        self.bio_emb_size: int
        self.train_batch: int
        self.eval_batch: int
        self.ssl_batch: int
        self.embedding_type: str
        self.global_relation_prediction: int
        self.grp_type: str

        self.__dict__ = json.load(open(path, 'r'))
        
    def save_hyperparameters(self):
        json.dump(self.__dict__, open(os.path.join(self.model_dir, 'hyperparameters.json'), 'w'))
        
