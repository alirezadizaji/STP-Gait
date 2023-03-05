from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple
from typing_extensions import Protocol

from dig.xgraph.models.models import GCNConv
from torch import nn
from torch_geometric.data import Batch, Data
import torch

from ...context import Skeleton
from ...models.gcn_lstm_transformer import LSTMConfig, GCNLayerConfig, Protocol1, GCNLayer

class Model(nn.Module):
    def __init__(self,
            n: int = 2,
            lstm_conf: LSTMConfig = LSTMConfig(), 
            gcn_conf: GCNLayerConfig = GCNLayerConfig(),
            get_gcn_edges: Callable[[int], torch.Tensor] = lambda T: Skeleton.get_interframe_edges_mode2(T, I=30, offset=20),
            init_lstm_hidden_state: Protocol1 = lambda lstm_num_layer, batch_size, hidden_size: torch.randn(lstm_num_layer, batch_size, hidden_size)) -> None:

        super().__init__()
        self.gcn_conf = gcn_conf.__dict__
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

    def _get_lstm_hidden_state(self, lstm_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hs[lstm_idx] is None:
            self.hs[lstm_idx] = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])
            self.cs[lstm_idx] = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])

        return self.hs[lstm_idx], self.cs[lstm_idx]

    def _update_lstm_hidden_state(self, lstm_idx: int, h: torch.Tensor, c: torch.Tensor) -> None:
        self.hs[lstm_idx] = h
        self.cs[lstm_idx] = c

    def _calc_edge_weight(self, node_valid: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
        if node_valid is None:
            return None
        
        row, col = self.edge_index
        row_valid, col_valid = node_valid[row], node_valid[col]
        edge_valid = torch.logical_and(row_valid, col_valid).float()
        return edge_valid

    def forward(self, x: torch.Tensor, x_valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                data = Batch.from_data_list([Data(x=x_.reshape(-1, D), edge_index=self.edge_index, edge_weight=self._calc_edge_weight(xv_)) for x_, xv_ in zip(x1, x_valid)])
            else:
                data = Batch.from_data_list([Data(x=x_.reshape(-1, D), edge_index=self.edge_index) for x_ in x1])
            x1 = gcn(data)

            data.x = x1
            datas: List[Data] = Batch.to_data_list(data)
            x1 = torch.stack([d.x for d in datas])
            x1 = x1.reshape(N, T, -1, D).reshape(N, T, -1)

        x1 = x1.reshape(N, T, -1, D)
        return x1