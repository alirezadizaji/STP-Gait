from torch import nn
from torch.nn import functional as F
import torch

from ..others.st_gcn.utils.graph import Graph
from ..others.st_gcn.st_gcn import st_gcn 

class Model1(nn.Module):
    def __init__(self, in_channels, num_class, edge_importance_weighting, graph_args,
                 **kwargs):
        super(Model1, self).__init__()

        # load graph
        if graph_args is None:
            graph_args = {}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 60, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(60, 60, kernel_size, 2, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        # N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2)[..., None]
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = x.permute(0, 2, 3, 1)
        return x


class Model2(nn.Module):
    def __init__(self, A, num_class, edge_importance_weighting, graph_args, **kwargs):
        super(Model2, self).__init__()

        self.A = A
        
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(60, 60, kernel_size, 2, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(60, num_class, kernel_size=1)

    def forward(self, x):
        # N, T, V, C 
        x = x.permute(0, 3, 1, 2)[..., None] 

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3)
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A.to(x.device) * importance.to(x.device))

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x