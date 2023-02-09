""" ViViT Model implementation, paper: https://arxiv.org/pdf/2103.15691.pdf """

from torch import nn
from torch import nn, Tensor

from .models.encoder import Encoder

class ViViT(nn.Module):
    def __init__(self, num_seq: int, num_classes: int, encoder: Encoder, dropout: int = 0.1):
        super().__init__()

        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(1),
            nn.Linear(num_seq * encoder.out_dim, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.fc(x)

        return x