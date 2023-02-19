from typing import List

from dig.xgraph.models import GCN_3l_BN, GCNConv, GlobalMeanPool
import torch
from torch import nn, Tensor
from torch_geometric.data import Batch, Data

class _GCNConv(nn.Module):
    def __init__(self, d1: int, d2: int):
        super().__init__()

        self.conv = GCNConv(d1, d2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(d2),
            nn.ReLU()
        )

    def forward(self, x: Tensor, edge_index: Tensor):
        return self.bn_relu(self.conv(x, edge_index))

class _GCNConvFC(_GCNConv):
    def __init__(self, d: int, num_classes: int, dropout_p: float = 0.2):
        super().__init__(d, d)
        self.pool = GlobalMeanPool()
        self.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(d, num_classes),
                nn.LogSoftmax(dim=1),
        )
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        x = super().forward(x, edge_index)
        x = self.pool(x, batch)
        x = self.fc(x)

        return x

class GCNCCT(nn.Module):
    def __init__(self, num_classes: int, num_shared_gcn: int, dim_node: int, dim_hidden: int, num_aux_cls: int = 8):
        super().__init__()
        
        self.shared: nn.ModuleList = nn.ModuleList(
            [_GCNConv(dim_node, dim_hidden)] +
            [_GCNConv(dim_hidden, dim_hidden) for _ in range(num_shared_gcn - 1)]
        )

        self.main = _GCNConvFC(dim_hidden, num_classes)
        self.auxs: nn.ModuleList = nn.ModuleList(
            [_GCNConvFC(dim_hidden, num_classes) for _ in range(num_aux_cls)])

        self._num_splits: int = num_aux_cls + 1

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        B, T, V, _ = x.size()

        x = x.flatten(1, -2) # N, T*V, D
        data = Batch.from_data_list([Data(x=x_, edge_index=edge_index) for x_ in x])
        x = data.x

        window_size = T // self._num_splits
        si = window_size * (1 + torch.arange(self._num_splits - 1))
        fi = torch.arange(T, dtype=torch.int64).repeat(2)

        for s in self.shared:
            x = s(x, data.edge_index)

        o_main = self.main(x, data.edge_index, data.batch)        # B, C
        x = x.reshape(B, T, V, -1)

        o_auxs: List[torch.Tensor] = list()
        for i in range(self._num_splits - 1):
            idx = fi[si[i]: si[i] + T]

            o_aux = self.auxs[i](x[:, idx].flatten(0, -2), data.edge_index, data.batch)
            o_auxs.append(o_aux)

        o_aux = torch.stack(o_auxs, dim=1)              # B, A, C

        return o_main, o_aux