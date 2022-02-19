import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD
from pytorch_transformers import AdamW, WarmupLinearSchedule

from lib.preprocessings import ScienceIE_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader, Unlabelled_Selection_Dataset, Unlabelled_Selection_loader
from lib.metrics import F1_triplet, F1_ner
from lib.models import MultiHeadSelection
from lib.config import Hyper


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='conll_selection_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|ssl|evaluation')
parser.add_argument('--bath_accumulation_size',
                    '-bms',
                    type=str,
                    default='0')
args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.hyper = Hyper(os.path.join('experiments', self.exp_name + '.json'))
        self.model_dir = self.hyper.model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("created directory: ", self.model_dir)
        else:
            print("directory: ", self.model_dir, " exits")
        
        self.hyper.save_hyperparameters()
        self.uda_lambda = 1
        self.embedding_type = self.hyper.embedding_type
        self.gpu = self.hyper.gpu
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.preprocessor = None
        self.optimizer = None
        self.learning_rate = self.hyper.learning_rate
        self.model = None
        self.performance = 0
        self.bath_accumulation_size = int(args.bath_accumulation_size)

    def _optimizer(self, name, model, learning_rate):
        if self.hyper.cell_name == 'bert':
            
            # have to specify different learning rate for BERT model and other parameters
            bert_params_name = list(filter(lambda kv: 'bert_embeddings' in kv[0], model.named_parameters()))
            base_params_name = list(filter(lambda kv: 'bert_embeddings' not in kv[0], model.named_parameters()))
            
            bert_params = list(map(lambda x: x[1], bert_params_name))
            base_params = list(map(lambda x: x[1], base_params_name))
            
            return Adam([
                            {'params': bert_params },
                            {'params': base_params, 'lr': learning_rate}
                        ], lr=2e-5)
        else:
            return Adam(model.parameters(), lr=learning_rate)

    def _init_model(self):
        self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)

    def preprocessing(self):
        if self.exp_name == 'conll_selection_re':
            self.preprocessor = ScienceIE_selection_preprocessing(self.hyper)
        elif self.exp_name == 'scienceie_selection_re':
            self.preprocessor = ScienceIE_selection_preprocessing(self.hyper)
        elif self.exp_name == 'ade_selection_re':
            self.preprocessor = ScienceIE_selection_preprocessing(self.hyper)
        
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model, self.learning_rate)
            if self.bath_accumulation_size > 0:
                self.ba_train()
            else:
                self.train()
            self.test()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model()
            self.test()
        elif mode == 'ssl':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model, self.learning_rate)
            if self.bath_accumulation_size > 0:
                self.ssl_ba_train()
            else:
                self.ssl_train()
            self.test()
        else:
            raise ValueError('invalid mode')

    def load_model(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,'best_model.pt')))

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, 'best_model.pt'))
        
    def test(self, write_result=True, save_model=False):
        test_set = Selection_Dataset(self.hyper, self.hyper.test)
        loader = Selection_loader(test_set, batch_size=self.hyper.eval_batch, pin_memory=False)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            to_print = 'Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ])
            
            self.record_test(to_print)
            print(to_print)

    def evaluation(self, epoch, write_result=True, save_model=False):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=False)
        self.triplet_metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            to_print = 'Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ])
            
            if save_model and triplet_result['fscore'] > self.performance:
                self.save_model(epoch)
                self.performance = triplet_result['fscore']
            if write_result:
                self.record_evaluation(epoch, to_print, path="scienceie_char_glove100d_unfreeze_enhanced_test")
            else:
                self.record_evaluation(epoch, to_print)
            print(to_print)

    
    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=False)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.record_loss(epoch, output['description'](epoch, self.hyper.epoch_num))

            if epoch % self.hyper.print_epoch == 0:
                self.evaluation(epoch, write_result=False, save_model=True)
                
    def ba_train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=False)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            self.optimizer.zero_grad()
            for batch_idx, sample in pbar:
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss /= self.bath_accumulation_size
                loss.backward()
                
                if (batch_idx + 1) % self.bath_accumulation_size == 0 or batch_idx == len(loader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.record_loss(epoch, output['description'](epoch, self.hyper.epoch_num))

            if epoch % self.hyper.print_epoch == 0:
                self.evaluation(epoch, write_result=False, save_model=True)
                
    def ssl_ba_train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=False)
        unlabelled_set = Unlabelled_Selection_Dataset(self.hyper, "unlabelled_0.1_more.txt", "unlabelled_0.5_more.txt")
        unlabelled_loader = Unlabelled_Selection_loader(unlabelled_set, batch_size=self.hyper.ssl_batch, pin_memory=False)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))
            unlabelled_batches = iter(unlabelled_loader)
            
            self.optimizer.zero_grad()
            for batch_idx, sample in pbar:
                output = self.model(sample, is_train=True)
                loss = output['loss']
                
                try:
                    unlabelled_batch = next(unlabelled_batches)
                    ssl_loss = self._ssl_train(unlabelled_batch)
                    loss += self.uda_lambda * ssl_loss
                except StopIteration:
                    print("Not using unlabelled data anymore on batch:", batch_idx)
                
                loss /= self.bath_accumulation_size    
                loss.backward()                
                if (batch_idx + 1) % self.bath_accumulation_size == 0 or batch_idx == len(loader) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.record_loss(epoch, output['description'](epoch, self.hyper.epoch_num))

            if epoch % self.hyper.print_epoch == 0:
                self.evaluation(epoch, write_result=False, save_model=True)
    
    def ssl_train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=False)
        unlabelled_set = Unlabelled_Selection_Dataset(self.hyper, "unlabelled_0.1_more.txt", "unlabelled_0.5_more.txt")
        unlabelled_loader = Unlabelled_Selection_loader(unlabelled_set, batch_size=self.hyper.ssl_batch, pin_memory=False)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))
            unlabelled_batches = iter(unlabelled_loader)

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                
                try:
                    unlabelled_batch = next(unlabelled_batches)
                    ssl_loss = self._ssl_train(unlabelled_batch)
                    loss += self.uda_lambda * ssl_loss
                except StopIteration:
                    print("Not using unlabelled data anymore on batch:", batch_idx)
                    
                loss.backward()                
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.record_loss(epoch, output['description'](epoch, self.hyper.epoch_num))

            if epoch % self.hyper.print_epoch == 0:
                self.evaluation(epoch, write_result=False, save_model=True)
                
    def _ssl_train(self, sample):
        # pseudo label from weakly augmented
        sample.text = sample.dataset1
        sample.tokens = sample.tokens1
        with torch.no_grad():
            output = self.model(sample, is_train=False, is_inference=True)
        ssl_bio_tag = torch.tensor(output['predicted_tag']).cuda(self.hyper.gpu)
        ssl_bio_mask = output['bio_mask']
        ssl_selection_tag = output['selection_gold_logits']
        ssl_selection_mask = output['selection_mask']
        ssl_grp_tag = output['grp_gold_logits']
        
        # logits from strongly augmented
        sample.text = sample.dataset2
        sample.tokens = sample.tokens2
        output = self.model(sample, is_train=False, is_inference=True)
        ssl_emission = output['emission_tag']
        ssl_selection_logits = output['selection_logits']
        ssl_grp_logits = output['grp_logits']
        
        # compute ssl loss
        ssl_crf_loss = -self.model.tagger(ssl_emission, ssl_bio_tag, mask=ssl_bio_mask, reduction='mean')
        ssl_selection_loss = self.masked_BCEloss(ssl_selection_mask, ssl_selection_logits, ssl_selection_tag)
        ssl_grp_loss = nn.CrossEntropyLoss()(ssl_grp_logits.view(-1, 2), ssl_grp_tag.view(-1))
        ssl_loss = ssl_crf_loss + ssl_selection_loss + ssl_grp_loss
        return ssl_loss
            
    
    def record_loss(self, epoch, losses):
        path = os.path.join(self.model_dir, self.exp_name + '_' + str(epoch) + 'loss.txt')
        with open(path, "w") as f:
            f.write(losses + '\n')
        f.close()
    
    def record_test(self, test):
        path = os.path.join(self.model_dir, self.exp_name + '_test.txt')
        with open(path, "w") as f:
            f.write(test + '\n')
        f.close()
    
    def record_evaluation(self, epoch, evaluation, path=None):
        if path is None:
            path = os.path.join(self.model_dir, self.exp_name + '_' + str(epoch) + 'evaluation.txt')
        else:
            path = os.path.join(path, self.exp_name + '_' + str(epoch) + 'evaluation.txt')
        with open(path, "w") as f:
            f.write(evaluation + '\n')
        f.close()
        
    def masked_BCEloss(self, selection_mask, selection_logits, selection_gold):
        # convert true/false to 1/0
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= selection_mask.sum() # here is different
        return selection_loss


if __name__ == "__main__":
    main = Runner(exp_name=args.exp_name)
    main.run(mode=args.mode)
