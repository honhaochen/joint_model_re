from typing import Any, Dict

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
            self,
            encoding_dim: int,
            num_classes: int,
            dropout_prob: float = 0.1,
    ):
        super(Classifier, self).__init__()
        self.encoding_dim = encoding_dim
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.dropout_layer = nn.Dropout(self.dropout_prob)
        self.classification_layer = nn.Linear(self.encoding_dim, self.num_classes, bias=True)

    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        encoding = self.dropout_layer(inputs)
        logits = self.classification_layer(encoding)
        return logits