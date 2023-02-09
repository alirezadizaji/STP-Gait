""" ViViT Model implementation, paper: https://arxiv.org/pdf/2103.15691.pdf """

from torch import nn
from torch import nn, Tensor

from .models.encoder import Encoder

class ViViT(nn.Module):
    def __init__(self, num_classes: int, encoder: Encoder, dropout: int = 0.2):
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.out_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.fc(x)

        return x