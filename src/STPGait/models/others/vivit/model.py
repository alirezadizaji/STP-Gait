""" ViViT Model implementation, paper: https://arxiv.org/pdf/2103.15691.pdf """

import torch
from torch import nn, Tensor

from .models.encoder import Encoder

class ViViT(nn.Module):
    def __init__(self, num_classes: int, encoder: Encoder, dropout: int = 0.1):
        super().__init__()

        self.encoder = encoder
        self.pe: nn.Parameter = None

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.out_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        _, n_temporal, n_spatial, d = x.size()
        if self.pe is None:
            self.pe = nn.Parameter(torch.randn(1, n_temporal, n_spatial, d)).to(x.device)
        
        x = x + self.pe
        x = self.encoder(x)
        
        if x.ndim == 4:
            d1, *_, d4 = x.size()
            x = x.reshape(d1, -1 , d4)

        x = torch.mean(x, dim=1)
        x = self.fc(x)

        return x