from dataclasses import dataclass
from typing import Callable, Tuple

from dig.xgraph.models.models import GCNConv
from torch import nn
import torch

from ..context import Skeleton

@dataclass
class LSTMConfig:
    input_size: int = 50
    hidden_size: int = 50
    num_layers: int = 1
    batch_first: bool = True

@dataclass
class GCNLayerConfig:
    hidden_size: int = 20
    dim_node: int = 2

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
    def __init__(self,hidden_size: int = 20, dim_node: int = 2):
        super().__init__()

        self.conv = GCNConv(dim_node, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_size, dim_node)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, T, L = x.size()
        # hard code
        V = L // 2
        x = x.reshape(N, T, V, 2)
        x = x.reshape(N*T*V, 2)

        x = self.conv(x, edge_index=edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index=edge_index)
        
        x = x.reshape(N, T, V, 2)
        x = x.reshape(N, T, L)
        return x

class GCNLSTMTransformer(nn.Module):
    def __init__(self,
            num_classes: int = 6,
            n: int = 2,
            fc_hidden: int = 50,
            lstm_conf: LSTMConfig = LSTMConfig(), 
            gcn_conf: GCNLayerConfig = GCNLayerConfig(),
            cnn_conf: CNNConf = CNNConf(),
            transformer_encoder_conf: TransformerEncoderConf = TransformerEncoderConf(),
            loss1: Callable[..., torch.Tensor] = lambda x, y: nn.MSELoss()(x, y),
            loss2: Callable[..., torch.Tensor] = lambda x, y: - torch.mean(x[torch.arange(x.size(0)), y]),
            ratio_to_apply_loss1: float = 0.2) -> None:

        super().__init__()
        cnn_conf = cnn_conf.__dict__
        gcn_conf = gcn_conf.__dict__
        transformer_encoder_conf = transformer_encoder_conf.__dict__
        self.lstm_conf = lstm_conf.__dict__
        
        self.edge_index = None

        self.lstms: nn.ModuleList = nn.ModuleList()
        self.gcns: nn.ModuleList = nn.ModuleList()
        for _ in range(n):
            self.lstms.append(nn.LSTM(**self.lstm_conf))
            self.gcns.append(GCNLayer(**gcn_conf))

        self.ratio_to_apply_loss1: float = ratio_to_apply_loss1
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, fc_hidden)
        )
        self.loss1 = loss1

        self.conv = nn.Conv1d(**cnn_conf)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fc_hidden,
            nhead=transformer_encoder_conf['enc_n_heads'])        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=transformer_encoder_conf['n_enc_layers'], 
            norm=None)

        self.pool = nn.AdaptiveAvgPool1d((1,))
        self.fc_classfier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Flatten(1),
            nn.Linear(fc_hidden*cnn_conf["out_channels"], fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_classes),
            nn.LogSoftmax(dim=1),
        )

        self.loss2 = loss2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): N, T, F

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: prediction, loss1 (unsupervised), loss2 (supervised)
        """

        if self.edge_index is None:
            self.edge_index = Skeleton.get_interframe_edges_mode2(x.size(1), I=30, offset=20)

        x1 = x.clone()       
        for gcn, lstm in zip(self.gcns, self.lstms):
            h0 = torch.randn(self.lstm_conf["num_layers"], x.size(0), self.lstm_conf["hidden_size"]).to(x1.device)
            c0 = torch.randn(self.lstm_conf["num_layers"], x.size(0), self.lstm_conf["hidden_size"]).to(x1.device)

            x1, (_, _) = lstm(x1, (h0, c0)) 
            x1 = gcn(x=x1, edge_index=self.edge_index)

        # Unsupervised loss
        x_b1 = self.fc(x1)
        size = int(x_b1.size(1) * self.ratio_to_apply_loss1)
        idx1 = torch.arange(x_b1.size(0)).unsqueeze(1).repeat(1, size)
        idx2 = torch.randint(0, x_b1.size(1) - 1, size=(x_b1.size(0), size))
        loss1 = self.loss1(x_b1[idx1, idx2], x[idx1, idx2 + 1])

        x1 = self.conv(x1)
        x1 = self.encoder(x1)
        x1 = self.fc_classfier(x1)
        loss2 = self.loss2(x1, y)

        return x1.exp(), loss1, loss2