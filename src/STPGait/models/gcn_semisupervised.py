from typing import Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch.nn import functional as F

from ..context import Skeleton
from .gcn_3l_bn import GCN_3l_BN

def calc_edge_weight(edge_index: torch.Tensor, node_valid: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
    if node_valid is None:
        return None
    
    row, col = edge_index
    row_valid, col_valid = node_valid[row], node_valid[col]
    edge_valid = torch.logical_and(row_valid, col_valid).float()

    return edge_valid

class GCNSemiSupervised(nn.Module):
    def __init__(self, dim_node: int, dim_hidden: int, sup_num_classes: int, 
            unsup_num_classes: Optional[int] = None, part1: np.ndarray = Skeleton.LOWER_BODY, 
            part2: np.ndarray = Skeleton.UPPER_BODY) -> None:
        super().__init__()

        self.p1_idx = part1
        self.p2_idx = part2
        
        if unsup_num_classes is None:
            unsup_num_classes = sup_num_classes

        self.encoder = nn.Linear(dim_node, dim_hidden)

        # Supervised branch
        self.gcn_supervised = GCN_3l_BN(model_level="graph", dim_node=dim_hidden, 
            dim_hidden=dim_hidden, num_classes=sup_num_classes)
        
        # Unsupervised branch
        self.gcn_unsupervised_lower = GCN_3l_BN(model_level="graph", dim_node=dim_hidden,
            dim_hidden=dim_hidden, num_classes=unsup_num_classes)
        self.gcn_unsupervised_upper = GCN_3l_BN(model_level="graph", dim_node=dim_hidden,
            dim_hidden=dim_hidden, num_classes=unsup_num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_index_p1: torch.Tensor, 
            edge_index_p2: torch.Tensor, y: torch.Tensor, node_invalid: torch.Tensor, 
            labeled: torch.Tensor):

        assert node_invalid.ndim == 3, "Node invalid must have N, T, V shape."
        x = self.encoder(x)                         
        x1 = x[:, :, self.p1_idx]
        niv1 = node_invalid[:, :, self.p1_idx].flatten(1)

        x2 = x[:, :, self.p2_idx]
        niv2 = node_invalid[:, :, self.p2_idx].flatten(1)

        niv = node_invalid.flatten(1)

        if torch.any(labeled):
            data1 = Batch.from_data_list([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index, 
                edge_weight=calc_edge_weight(edge_index, ni)) for x_, ni in zip(x[labeled], ~niv[labeled])])
            
            o_sup = self.gcn_supervised(x=data1.x, edge_index=data1.edge_index, batch=data1.batch, edge_weight=data1.edge_weight)
            
            olog_sup = F.log_softmax(o_sup)
            idx = torch.arange(olog_sup.size(0))
            sup_loss = -torch.mean(olog_sup[idx, y[labeled]])
            y_pred = olog_sup.argmax(1)
        else:
            y_pred = None
            sup_loss = None

        # Unsupervised forwarding
        ## 1st: Forward first part of body; e.g. lower part
        data2 = Batch.from_data_list([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index_p1, 
            edge_weight=calc_edge_weight(edge_index_p1, nv)) for x_, nv in zip(x1, ~niv1)])
        o1_unsup = self.gcn_unsupervised_lower(x=data2.x, edge_index=data2.edge_index, batch=data2.batch, edge_weight=data2.edge_weight)

        ## 2nd: Forward 2nd part of body; e.g. upper part
        data3 = Batch.from_data_list([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index_p2, 
            edge_weight=calc_edge_weight(edge_index_p2, nv)) for x_, nv in zip(x2, ~niv2)])
        o2_unsup = self.gcn_unsupervised_upper(x=data3.x, edge_index=data3.edge_index, batch=data3.batch, edge_weight=data3.edge_weight)


        o1log_unsup = F.log_softmax(o1_unsup)
        o2log_unsup = F.log_softmax(o2_unsup)
        
        y1 = o1log_unsup.argmax(1).detach()
        y2 = o2log_unsup.argmax(1).detach()
        idx = torch.arange(y1.numel())
        u1_loss = -torch.mean(o1log_unsup[idx, y2])
        u2_loss = -torch.mean(o2log_unsup[idx, y1])
        
        return y_pred, y1, y2, sup_loss, u1_loss, u2_loss