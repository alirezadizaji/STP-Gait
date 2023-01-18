from dataclasses import dataclass
from typing import Callable, Tuple

from dig.xgraph.models import GCNConv
from torch import nn
import torch

@dataclass
class LSTMConfig:
    input_size: int = 50
    hidden_size: int = 50
    num_layers: int = 1
    batch_first: bool = True

@dataclass
class GCNLayerConfig:
    hidden_size: int = 50
    dim_node: int = 25

@dataclass
class CNNConf:
    in_channels: int = 417
    out_channels: int = 85
    kernel_size: int = 1
    stride: int = 1

@dataclass
class TransformerEncoderConf:
    enc_n_heads: int = 5
    n_enc_layers: int = 3

class GCNLayer(nn.Module):
    def __init__(self, hidden_size: int = 50, dim_node: int = 25):
        super().__init__()

        self.conv = GCNConv(dim_node, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x) -> torch.Tensor:
        return self.relu(self.conv(self.bn(x)))

class GCNLSTMTransformer(nn.Module):
    def __init__(self,
            n: int = 2,
            lstm_conf: LSTMConfig = LSTMConfig(), 
            gcn_conf: GCNLayerConfig = GCNLayerConfig(),
            cnn_conf: CNNConf = CNNConf(),
            transformer_encoder_conf: TransformerEncoderConf = TransformerEncoderConf(),
            loss1: Callable[..., torch.Tensor] = lambda x, y: nn.MSELoss()(x, y),
            loss2: Callable[..., torch.Tensor] = lambda x, y: nn.MSELoss()(x, y),
            ratio_to_apply_loss1: float = 0.2) -> None:

        super().__init__()
        cnn_conf = cnn_conf.__dict__
        gcn_conf = gcn_conf.__dict__
        transformer_encoder_conf = transformer_encoder_conf.__dict__
        self.lstm_conf = lstm_conf.__dict__
        
        self.lstms: nn.ModuleList = nn.ModuleList()
        self.gcns: nn.ModuleList = nn.ModuleList()
        for _ in range(n):
            self.lstms.append(nn.LSTM(**self.lstm_conf))
            self.gcns.append(GCNLayer(**gcn_conf))

        self.ratio_to_apply_loss1: float = ratio_to_apply_loss1
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(gcn_conf['hidden_size'], gcn_conf['hidden_size'])
        )
        self.loss1 = loss1

        self.conv = nn.Conv1d(**cnn_conf)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gcn_conf['hidden_size'],
            nhead=transformer_encoder_conf['enc_n_heads'])        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_encoder_conf['n_enc_layers'], 
            norm=None)

        self.pool = nn.AdaptiveAvgPool1d((1,))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(gcn_conf['hidden_size'], gcn_conf['hidden_size']),
            nn.ReLU(),
            nn.Linear(gcn_conf['hidden_size'], gcn_conf['hidden_size']),
            nn.Softmax(dim=1),
        )

        self.loss2 = loss2

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): N, T, F

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: prediction, loss1 (unsupervised), loss2 (supervised)
        """
        x1 = x.clone()       
        for gcn, lstm in zip(self.gcns, self.lstms):
            h0 = torch.randn(self.lstm_conf["num_layers"], x.size(1), self.lstm_conf["hidden_size"])
            c0 = torch.randn(self.lstm_conf["num_layers"], x.size(1), self.lstm_conf["hidden_size"])
            x1, _, _ = lstm(x1, (h0, c0)) 
            x1 = gcn(x=x1, edge_index=edge_index)

        # Unsupervised loss
        x_b1 = self.fc(x1)
        size = int(x_b1.size(2) * self.ratio_to_apply_loss1)
        idx = torch.randperm(x_b1.size(2) - 1)[:size]
        loss1 = self.loss1(x_b1[idx], x[idx + 1])

        x1 = self.conv(x1)
        x1 = self.encoder(x1)
        x1 = self.fc(x1)
        loss2 = self.loss2(x1, y)

        return x1, loss1, loss2