import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from lib.modules import Classifier

from torchcrf import CRF
from pytorch_transformers import *
import numpy as np
from wasabi import Printer


class MultiHeadSelection(nn.Module):
    def __init__(self, hyper) -> None:
        super(MultiHeadSelection, self).__init__()
        
        self.msg_printer = Printer()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        self.max_length = hyper.max_text_len
        self.global_relation_prediction = hyper.global_relation_prediction
        self.grp_type = hyper.grp_type
        self.self_attention = hyper.self_attention
        
        self.relation_vocab = json.load(open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}
        self.word_vocab = json.load(open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.char_vocab = json.load(open(os.path.join(self.data_root, 'char_vocab.json'), 'r'))
        
        self.embedding_type = hyper.embedding_type
        if "glove" in self.embedding_type:
            print("Using GloVe word embeddings")
            self.embedding_filename = self.embedding_type.replace("_", ".") + "d.txt"
            self.glove_embeddings = self.load_glove_embedding()
            pre_trained_embeddings = self.get_embeddings_for_word_vocab(self.word_vocab)
            self.word_embeddings = nn.Embedding.from_pretrained(pre_trained_embeddings, freeze=False)
        elif "bert" in self.embedding_type:
            do_lower_case = "uncased" in self.embedding_type
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_type, 
                                                           do_lower_case=do_lower_case, add_special_tokens=False)
            self.bert_embeddings = AutoModel.from_pretrained(self.embedding_type, output_hidden_states=True)
        else:
            self.word_embeddings = nn.Embedding(num_embeddings=len(self.word_vocab), embedding_dim=hyper.emb_size)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
          
        # char_embedding
        if hyper.char_emb_size > 0:
            print("Using character BiLSTM embeddings")
            self.char_embeddings = nn.Embedding(len(self.char_vocab), hyper.char_emb_size)
            self.char_rnn = nn.LSTM(hyper.char_emb_size, 
                                    hyper.char_encoder_hidden_size, 
                                    bidirectional=True, 
                                    num_layers=1, 
                                    batch_first=True,)
        # --------------------
        
        self.relation_emb = nn.Embedding(num_embeddings=len(self.relation_vocab), embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab), embedding_dim=hyper.bio_emb_size)

        if hyper.cell_name == 'lstm':
            if hyper.char_emb_size > 0:
                self.encoder = nn.LSTM(hyper.emb_size + 2 * hyper.char_encoder_hidden_size,
                                       hyper.hidden_size,
                                       bidirectional=True,
                                       batch_first=True)
            elif "bert" in self.embedding_type:
                self.encoder = nn.LSTM(768,
                                       hyper.hidden_size,
                                       bidirectional=True,
                                       batch_first=True)
            else:
                self.encoder = nn.LSTM(hyper.emb_size,
                                       hyper.hidden_size,
                                       bidirectional=True,
                                       batch_first=True)
              
        elif hyper.cell_name == 'bert':
            self.encoder = nn.Linear(768, hyper.hidden_size)
        else:
            raise ValueError('cell name should be lstm or bert!')
        
        # A linear layer from word embeddings to BIO for CRF tagging, times 2 due to bilstm
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)
        
        self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        
        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)
        
        # grp
        if hyper.cell_name == 'bert' and self.global_relation_prediction:
            self.grp = Classifier(encoding_dim=768,
                                  num_classes=2)
        # attention
        if self.self_attention:
            print("Using self_attention")
            self.self_attention_layer = nn.MultiheadAttention(hyper.hidden_size + hyper.bio_emb_size, 4, batch_first=True)
            

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
       
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold
        selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   selection_tags)
        return selection_triplets
    
    def selection_inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
       
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold
        
        batch_num = len(decoded_tag)
        result = [[] for _ in range(batch_num)]
        
        selection_outcome = torch.nonzero(selection_tags)
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }
        
        for i in range(selection_outcome.size(0)):
            b, s, p, o = selection_outcome[i].tolist()
            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue

            selection = {
                'object': o,
                'predicate': p,
                'subject': s
            }
            result[b].append(selection)
        return result
    
    def selection_gold_logits_inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
       
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold
        
        return selection_tags.float()

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        # convert true/false to 1/0
        selection_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab), -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output, grp):
        if grp:
            return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, L_grp: {:.2f}, epoch: {}/{}:".format(
                output['loss'].item(), output['crf_loss'].item(), output['selection_loss'].item(), output['grp_loss'].item(), 
                epoch, epoch_num)
        else:
            return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
                output['loss'].item(), output['crf_loss'].item(), output['selection_loss'].item(), 
                epoch, epoch_num)

    def forward(self, sample, is_train: bool, is_inference: bool = False) -> Dict[str, torch.Tensor]:
        if is_inference:
            text_list = sample.text
            tokens = sample.tokens.cuda(self.gpu)
            max_length = self.max_length
        
        else:
            tokens = sample.tokens_id.cuda(self.gpu)
            selection_gold = sample.selection_id.cuda(self.gpu)
            bio_gold = sample.bio_id.cuda(self.gpu)
            char_tokens = sample.char_tokens_id.cuda(self.gpu)
    
            text_list = sample.text
            spo_gold = sample.spo_gold
            bio_text = sample.bio
            max_length = self.max_length # max([len(lst) for lst in text_list]) + 2 
        
        if 'bert' in self.embedding_type:
            mask = tokens != self.word_vocab['<pad>'] # no mask seems every word has an embedding in bert
        else:
            mask = tokens != self.word_vocab['<pad>']
            
        bio_mask = mask

        if self.hyper.cell_name in ('lstm'):
            if self.hyper.char_emb_size > 0:
                batch_size, sentence_length, token_length = char_tokens.shape
                char_tokens = char_tokens.view(batch_size * sentence_length, token_length)
                
                # batch_size * max_line_length, max_token_length, char_emb_dim
                embedded_char_tokens = self.char_embeddings(char_tokens)
                # pass through bilstm

                # output: batch_size * max_line_length, max_token_length, num_directions * hidden_size
                # h_n = num_layers * num_directions, batch_size, hidden_dimension
                # c_n = num_layers * num_directions, batch_size, hidden_dimension
                output, (h_n, c_n) = self.char_rnn(embedded_char_tokens)
                
                # concat forward and backward hidden states
                forward_hidden = h_n[0, :, :]
                backward_hidden = h_n[1, :, :]
                char_encoding = torch.cat([forward_hidden, backward_hidden], dim=1)
                
                # batch_size, max_line_length, embedding_dimension
                char_encoding = char_encoding.view(batch_size, sentence_length, -1)  
                
                word_embedded = self.word_embeddings(tokens)
                concat_embedding = torch.cat([char_encoding, word_embedded], dim=2)
                
                o, h = self.encoder(concat_embedding)
                o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))
            
            elif "bert" in self.embedding_type:
                embeddings = []
                for text_lst in text_list:
                    sub_tokens_map = {}
                    total_subtokens = 0
                    for text in text_lst:
                        subtokens_length = len(self.tokenizer.tokenize(text))
                        sub_tokens_map[text] = subtokens_length
                        total_subtokens += subtokens_length

                    bert_tokens = self.tokenizer(text_lst, return_tensors="pt", is_split_into_words=True,
                                                 padding="max_length", truncation=True, max_length=max_length).to(self.gpu)  
                    padding_length = len(bert_tokens['input_ids'][0]) - total_subtokens - 2 # plus [CLS] and [SEP]
                    
                    with torch.no_grad():
                        bert_output = self.bert_embeddings(**bert_tokens) # 1, max_length, 768
                    sum_all_layers = sum(bert_output.hidden_states[1:13])[0]
                    if padding_length > 0:
                        sum_all_layers = sum_all_layers[:-padding_length]
                    sum_all_layers = sum_all_layers[1:len(sum_all_layers) - 1] # exclude [CLS] and [SEP]
                
                    index = 0
                    bert_embeddings = []
                    for text in text_lst:
                        bert_embedding = sum_all_layers[index]
                        for i in range(1, sub_tokens_map[text]):
                            bert_embedding += sum_all_layers[index + i]
                        bert_embedding /= sub_tokens_map[text]
                        bert_embeddings.append(bert_embedding)
                        index += sub_tokens_map[text]
                    
                    while len(bert_embeddings) < max_length:
                        zeros = torch.zeros(768).to(self.gpu)
                        bert_embeddings.append(zeros)
                    
                    embeddings.append(torch.stack(bert_embeddings))
                    
                embeddings = torch.stack(embeddings)
                o, h = self.encoder(embeddings)
                
                # divide by 2 due to bilstm layer
                o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))
            else:
                embedded = self.word_embeddings(tokens)
                o, h = self.encoder(embedded)
                
                # divide by 2 due to bilstm layer
                o = (lambda a: sum(a) / 2)(torch.split(o, self.hyper.hidden_size, dim=2))

        elif self.hyper.cell_name == 'bert':
            embeddings = []
            CLS_embeddings = []
            for text_lst in text_list:
                sub_tokens_map = {}
                total_subtokens = 0
                for text in text_lst:
                    subtokens_length = len(self.tokenizer.tokenize(text))
                    sub_tokens_map[text] = subtokens_length
                    total_subtokens += subtokens_length
                bert_tokens = self.tokenizer(text_lst, return_tensors="pt", is_split_into_words=True,
                                             padding="max_length", truncation=True, max_length=max_length).to(self.gpu)  
                padding_length = len(bert_tokens['input_ids'][0]) - total_subtokens - 2 # plus [CLS] and [SEP]
                
                # with torch.no_grad():
                bert_output = self.bert_embeddings(**bert_tokens) # 1, max_length, 768
                
                sum_all_layers = sum(bert_output.hidden_states[1:13])[0]
                if padding_length > 0:
                    sum_all_layers = sum_all_layers[:-padding_length]
                CLS_embedding = sum_all_layers[0]
                sum_all_layers = sum_all_layers[1:len(sum_all_layers) - 1] # exclude [CLS] and [SEP]
                if self.grp_type == "mean":
                    CLS_embedding = (CLS_embedding + sum(sum_all_layers)) / (len(sum_all_layers) + 1)
            
                index = 0
                bert_embeddings = []
                for text in text_lst:
                    bert_embedding = sum_all_layers[index]
                    for i in range(1, sub_tokens_map[text]):
                        bert_embedding += sum_all_layers[index + i]
                    bert_embedding /= sub_tokens_map[text]
                    bert_embeddings.append(bert_embedding)
                    index += sub_tokens_map[text]
                
                while len(bert_embeddings) < max_length:
                    zeros = torch.zeros(768).to(self.gpu)
                    bert_embeddings.append(zeros)
                
                CLS_embeddings.append(CLS_embedding)
                embeddings.append(torch.stack(bert_embeddings))
            
            CLS_embeddings = torch.stack(CLS_embeddings)
            embeddings = torch.stack(embeddings)
            o = self.encoder(embeddings)
            
            if is_train and self.global_relation_prediction:
                # process text_list
                grp_labels = []
                for spo in spo_gold:
                    if len(spo) > 0:
                        grp_labels.append(1)
                    else:
                        grp_labels.append(0)
                grp_labels = torch.tensor(grp_labels).to(self.gpu)
                grp_logits = self.grp(CLS_embeddings)

        else:
            raise ValueError('unexpected encoder name!')
        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
        else:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)
            
            if is_inference:
                output['predicted_tag'] = self.tagger.decode(emissions=emi)
                output['emission_tag'] = emi
                output['bio_mask'] = bio_mask
            if not is_inference:
                output['gold_tags'] = bio_text
                output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            bio_gold = torch.tensor(temp_tag).cuda(self.gpu)
        
        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)
        
        # can add self-attention layer here
        if self.self_attention:
            o, _ = self.self_attention_layer(o, o, o)
        
        # forward multi head selection
        B, L, H = o.size()
        
        # duplicate sentences
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        
        # duplicate words
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        
        # concate sentences with words
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # bijh = batch, # of tokens, # of tokens, # embeddings
        # rh = # of embedding, # embeddings
        # birj = batch, # of tokens, # of embedding, # of tokens
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        if not is_train:
            output['selection_triplets'] = self.inference(mask, text_list, decoded_tag, selection_logits)
            # output['selection_list'] = self.selection_inference(mask, text_list, decoded_tag, selection_logits)
            output['selection_logits'] = selection_logits
            output['selection_gold_logits'] = self.selection_gold_logits_inference(mask, text_list, decoded_tag, selection_logits)
            output['selection_mask'] =(mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(2).expand(-1, -1, len(self.relation_vocab), -1)
            
            if is_inference and self.global_relation_prediction:
                # process text_list
                grp_labels = []
                for spo in output['selection_triplets']:
                    if len(spo) > 0:
                        grp_labels.append(1)
                    else:
                        grp_labels.append(0)
                grp_labels = torch.tensor(grp_labels).to(self.gpu)
                grp_logits = self.grp(CLS_embeddings)
                output['grp_logits'] = grp_logits
                output['grp_gold_logits'] = grp_labels
            
            if not is_inference:
                output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits, selection_gold)
        
        grp_loss = 0
        if is_train and self.global_relation_prediction:
            grp_loss = nn.CrossEntropyLoss()(grp_logits.view(-1, 2), grp_labels.view(-1))
            output['grp_loss'] = grp_loss
        
        loss = crf_loss + selection_loss
        if self.global_relation_prediction:
            loss += grp_loss
        
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['grp_loss'] = grp_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output, grp=self.global_relation_prediction)
        return output
    
    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return entity

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        
        # !!! important operation, getting indices of non-zeros in the logits
        idx = torch.nonzero(selection_tags.cpu())
        
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()
            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)

            assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)
        return result
    
    def load_glove_embedding(self) -> Dict[str, np.array]:
        """
        Imports the glove embedding
        Loads the word embedding for words in the vocabulary
        If the word in the vocabulary does not have an embedding
        then it is loaded with zeros
        """
        embedding_dim = int(self.embedding_type.split("_")[-1])
        self.embedding_dimension = embedding_dim
        glove_embeddings: Dict[str, np.array] = {}
        with self.msg_printer.loading("Loading GLOVE embeddings"):
            with open(self.embedding_filename, "r") as fp:
                for line in fp:
                    values = line.split()
                    word = values[0]
                    embedding = np.array([float(value) for value in values[1:]])
                    glove_embeddings[word] = embedding

        return glove_embeddings
    
    def get_embeddings_for_word_vocab(self, item2idx) -> torch.FloatTensor:
        idx2item = {}
        for item, idx in item2idx.items():
            idx2item[idx] = item
        
        len_vocab = len(idx2item)
        embeddings = []
        for idx in range(len_vocab):
            item = idx2item.get(idx)
            if item == "oov":
                emb = np.zeros(self.hyper.emb_size)
            else:
                try: 
                    # try getting the embeddings from the embeddings dictionary
                    emb = self.glove_embeddings[item]
                except KeyError:
                    try:
                        # try lowercasing the item and getting the embedding
                        emb = self.glove_embeddings[item.lower()]
                    except KeyError:
                        # nothing is working, lets fill it with random integers from normal dist
                        emb = np.random.randn(self.hyper.emb_size)
            embeddings.append(emb)

        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return embeddings