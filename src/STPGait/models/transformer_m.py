from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from torch import nn
import torch

@dataclass
class TransformerEncoderConf:
    enc_n_heads: int = 5
    n_enc_layers: int = 3

class Transformer(nn.Module):
    def __init__(self,
            num_classes: int = 6,
            fc_hidden: int = 50,
            transformer_encoder_conf: Optional[TransformerEncoderConf] = TransformerEncoderConf(),
            loss_func: Callable[..., torch.Tensor] = lambda x, y: - torch.mean(x[torch.arange(x.size(0)), y]),
            ) -> None:

        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, fc_hidden)
        )
        self.loss_func = loss_func

        transformer_encoder_conf = transformer_encoder_conf.__dict__
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fc_hidden,
            nhead=transformer_encoder_conf['enc_n_heads'])        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_encoder_conf['n_enc_layers'], 
            norm=None)

        self.fc_classfier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(1),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
            nn.LogSoftmax(dim=1),
        )


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        # Use only first sequence
        x = x[:, [0]]

        x = self.fc_classfier(x)
        loss = self.loss_func(x, y)

        return x.exp(), loss