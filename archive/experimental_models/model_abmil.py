"""Archived attention-based MIL baseline; not part of canonical MRePath."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import Attn_Net_Gated
from models.util import initialize_weights


class ABMIL(nn.Module):
    def __init__(
        self,
        path_input_dim: int = 1024,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.25,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.instance_encoder = nn.Sequential(
            nn.Linear(path_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = Attn_Net_Gated(
            L=hidden_dim,
            D=attention_dim,
            dropout=dropout > 0,
            n_classes=1,
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.apply(initialize_weights)

    def forward(self, data_WSI, data_omics=None, mask=None, **kwargs):
        del data_omics, mask, kwargs
        if data_WSI.ndim == 3:
            if data_WSI.shape[0] != 1:
                raise ValueError("ABMIL requires batch_size=1")
            data_WSI = data_WSI.squeeze(0)
        if data_WSI.ndim != 2:
            raise ValueError(
                f"Expected [patches, dim], got {tuple(data_WSI.shape)}"
            )
        instances = self.instance_encoder(data_WSI.float())
        logits, _ = self.attention(instances)
        scores = F.softmax(logits.transpose(1, 0), dim=1)
        return self.classifier(torch.mm(scores, instances))
