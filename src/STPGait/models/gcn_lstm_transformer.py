from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
from typing_extensions import Protocol

from dig.xgraph.models.models import GCNConv
from torch import nn
from torch_geometric.data import Batch, Data
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
    
    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        x = self.conv(x=x, edge_index=edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x=x, edge_index=edge_index)

        return x

class Protocol1(Protocol):
    def __call__(self, num_layers: int, batch_size: int, hidden_size: int) -> torch.Tensor:
        ...

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
            ratio_to_apply_loss1: float = 0.2,
            get_gcn_edges: Callable[[int], torch.Tensor] = lambda T: Skeleton.get_interframe_edges_mode2(T, I=30, offset=20),
            init_lstm_hidden_state: Protocol1 = lambda lstm_num_layer, batch_size, hidden_size: torch.randn(lstm_num_layer, batch_size, hidden_size)) -> None:

        super().__init__()
        cnn_conf = cnn_conf.__dict__
        self.gcn_conf = gcn_conf.__dict__
        transformer_encoder_conf = transformer_encoder_conf.__dict__
        self.lstm_conf = lstm_conf.__dict__
        
        self.get_gcn_edges = get_gcn_edges
        self.edge_index = None
        self.init_lstm_hidden_state = init_lstm_hidden_state

        self.lstms: nn.ModuleList = nn.ModuleList()
        self.hs: List[torch.Tensor] = list()
        self.cs: List[torch.Tensor] = list()

        self.gcns: nn.ModuleList = nn.ModuleList()
        
        for _ in range(n):
            self.lstms.append(nn.LSTM(**self.lstm_conf))
            self.hs.append(None)
            self.cs.append(None)

            self.gcns.append(GCNLayer(**self.gcn_conf))

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

    def _get_lstm_hidden_state(self, lstm_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])
        c0 = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])

        return h0, c0

    def _update_lstm_hidden_state(self, lstm_idx: int, h: torch.Tensor, c: torch.Tensor) -> None:
        pass

    def _calc_edge_attr(self, node_valid: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
        if node_valid is None:
            return None
        
        row, col = self.edge_index
        row_valid, col_valid = node_valid[row], node_valid[col]
        edge_valid = torch.logical_and(row_valid, col_valid).long()
        return edge_valid

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): N, T, V*D
            x_valid (torch.Tensor): A boolean tensor; If True then that node is valid, O.W. invalid (shape: N, T*V)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: prediction, loss1 (unsupervised), loss2 (supervised)
        """

        if self.edge_index is None:
            self.edge_index = self.get_gcn_edges(x.size(1)).to(x.device)

        x1 = x.clone()       
        for idx, (gcn, lstm) in enumerate(zip(self.gcns, self.lstms)):
            h0, c0 = self._get_lstm_hidden_state(idx, x.size(0))
            h0 = h0.to(x1.device)
            c0 = c0.to(x1.device)

            x1, (h, c) = lstm(x1, (h0, c0)) 
            self._update_lstm_hidden_state(idx, h.detach(), c.detach())

            D = self.gcn_conf["dim_node"]
            N, T, _ = x1.size()
            x1 = x1.reshape(N, T, -1, D)
            
            if x_valid is not None:
                data = Batch.from_data_list([Data(x=x_.reshape(-1, D), edge_index=self.edge_index, edge_weight=self._calc_edge_attr(xv_)) for x_, xv_ in zip(x1, x_valid)])
            else:
                data = Batch.from_data_list([Data(x=x_.reshape(-1, D), edge_index=self.edge_index) for x_ in x1])
            x1 = gcn(data)

            data.x = x1
            datas: List[Data] = Batch.to_data_list(data)
            x1 = torch.stack([d.x for d in datas])
            x1 = x1.reshape(N, T, -1, D).reshape(N, T, -1)

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