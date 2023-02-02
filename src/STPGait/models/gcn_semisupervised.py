from typing import Optional

from dig.xgraph.models import GCN_3l_BN
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch.nn import functional as F

from ..context import Skeleton

def _calc_edge_weight(edge_index: torch.Tensor, node_valid: Optional[torch.Tensor]=None) -> Optional[torch.Tensor]:
    if node_valid is None:
        return None
    
    row, col = edge_index
    row_valid, col_valid = node_valid[row], node_valid[col]
    edge_valid = torch.logical_and(row_valid, col_valid).long()
    return edge_valid

class GCNSemiSupervised(nn.Module):
    def __init__(self, dim_node: int, dim_hidden: int, sup_num_classes: int, 
            unsup_num_classes: Optional[int] = None) -> None:
        super().__init_()

        if unsup_num_classes is None:
            unsup_num_classes = sup_num_classes

        self.encoder = nn.Linear(dim_node, dim_hidden)

        # Supervised branch
        self.gcn_supervised = GCN_3l_BN(model_level="graph", dim_node=dim_hidden, 
            num_classes=sup_num_classes)
        
        # Unsupervised branch
        self.gcn_unsupervised_lower = GCN_3l_BN(model_level="graph", dim_node=dim_hidden,
            num_classes=unsup_num_classes)
        self.gcn_unsupervised_upper = GCN_3l_BN(model_level="graph", dim_node=dim_hidden,
            num_classes=unsup_num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_index_upper: torch.Tensor, 
            edge_index_lower: torch.Tensor, y: torch.Tensor, node_invalid: torch.Tensor, 
            labeled: torch.Tensor):

        x = self.encoder(x)                          # N, T, V, D
        x_lower = x[:, :, Skeleton.LOWER_BODY]       # N, T, V1, D
        x_upper = x[:, :, Skeleton.UPPER_BODY]       # N, T, V2, D

        data1 = Batch([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index, 
            edge_weight=_calc_edge_weight(edge_index, ~node_invalid)) for x_ in x[labeled]])
        o_sup = self.gcn_supervised(data1)

        data2 = Batch([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index_lower, 
            edge_weight=_calc_edge_weight(edge_index, ~node_invalid)) for x_ in x_lower])
        data3 = Batch([Data(x=x_.flatten(end_dim=-2), edge_index=edge_index_upper, 
            edge_weight=_calc_edge_weight(edge_index, ~node_invalid)) for x_ in x_upper])
        ol_unsup = self.gcn_unsupervised_lower(data2)
        ou_unsup = self.gcn_unsupervised_upper(data3)

        olog_sup = F.log_softmax(o_sup)
        y_pred = olog_sup.argmax(1)
        ollog_unsup = F.log_softmax(ol_unsup)
        oulog_unsup = F.log_softmax(ou_unsup)

        idx = torch.arange(olog_sup.size(0))
        sup_loss = -torch.mean(olog_sup[idx, y[labeled]])

        yl = ollog_unsup.argmax(1).detach()
        yu = oulog_unsup.argmax(1).detach()
        idx = torch.arange(yl.numel())
        unsup_lower = -torch.mean(ollog_unsup[idx, yu])
        unsup_upper = -torch.mean(oulog_unsup[idx, yl])
        
        return y_pred, yl, yu, sup_loss, unsup_lower, unsup_upper