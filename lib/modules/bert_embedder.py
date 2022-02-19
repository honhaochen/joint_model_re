import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer

from typing import List, Union
import torch
import torch.nn as nn


class BertEmbedder(nn.Module):
    def __init__(
            self,
            embedding_type: str,
            tokenizer: nn.Module,
            gpu: str,
            grp_type: str = "mean",  
    ):
        super(BertEmbedder, self).__init__()
        self.embedding_type = embedding_type
        self.tokenizer = tokenizer
        self.gpu = gpu
        self.grp_type = grp_type
        self.bert_embeddings = AutoModel.from_pretrained(self.embedding_type, output_hidden_states=True)

    def forward(self, text_list: List[List[str]]) -> Union[torch.Tensor, torch.Tensor]:
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
        
        return embeddings, CLS_embeddings