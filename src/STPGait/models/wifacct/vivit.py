""" ViViT Encoder1 implementations """

import torch
from torch import nn, Tensor

from ..others.vivit import Encoder1

class Model1(Encoder1):
    def __init__(self, d_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, n_enc_layers, dim_feedforward, dropout, activation)
        self.pe: nn.Parameter = None
        
    def forward(self, x: Tensor) -> Tensor:
        B, nt, nv, d = x.size()
        if self.pe is None:
            self.pe = nn.Parameter(torch.randn(1, nt, nv, d)).to(x.device)
            
        x = x + self.pe
        x = super().forward(x)
    
        x = x.reshape(B, nt, nv, d)
        return x
 
class Model2(Encoder1):
    def __init__(self, num_classes, d_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, n_enc_layers, dim_feedforward, dropout, activation)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)

        x = torch.mean(x, dim=1)
        x = self.fc(x)

        return x