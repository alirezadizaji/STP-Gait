from torch import nn
from torch.nn import functional as F
import torch

import numpy as np
import math

import sys
# sys.path.append('../')
from ..others.mst_gcn.model.layers import Basic_Layer, Basic_TCN_layer, MS_TCN_layer, Temporal_Bottleneck_Layer, \
    MS_Temporal_Bottleneck_Layer, Temporal_Sep_Layer, Basic_GCN_layer, MS_GCN_layer, Spatial_Bottleneck_Layer, \
    MS_Spatial_Bottleneck_Layer, SpatialGraphCov, Spatial_Sep_Layer
from ..others.mst_gcn.model.activations import Activations
from ..others.mst_gcn.model.utils import import_class, conv_branch_init, conv_init, bn_init
from ..others.mst_gcn.model.attentions import Attention_Layer
from ..others.mst_gcn.graph.kinetics import Graph

__block_type__ = {
    'basic': (Basic_GCN_layer, Basic_TCN_layer),
    'bottle': (Spatial_Bottleneck_Layer, Temporal_Bottleneck_Layer),
    'sep': (Spatial_Sep_Layer, Temporal_Sep_Layer),
    'ms': (MS_GCN_layer, MS_TCN_layer),
    'ms_bottle': (MS_Spatial_Bottleneck_Layer, MS_Temporal_Bottleneck_Layer),
}

class Model1(nn.Module):
    def __init__(self, num_class, num_point, num_person, block_args, graph_args, kernel_size, block_type, atten,
                 **kwargs):
        super(Model1, self).__init__()

        kwargs['act'] = Activations(kwargs['act'])
        atten = None if atten == 'None' else atten
        
        self.graph = Graph(**graph_args)
        A = self.graph.A
        self.A = A

        self.data_bn = nn.BatchNorm1d(num_person * block_args[0][0] * num_point)

        self.layers = nn.ModuleList()

        for i, block in enumerate(block_args):
            if i == 0:
                self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                 kernel_size=kernel_size, stride=block[3], A=A, block_type='basic',
                                                 atten=None, **kwargs))
            else:
                self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                 kernel_size=kernel_size, stride=block[3], A=A, block_type=block_type,
                                                 atten=atten, **kwargs))

        for m in self.modules():
            if isinstance(m, SpatialGraphCov) or isinstance(m, Spatial_Sep_Layer):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        conv_branch_init(mm, self.graph.A.shape[0])
                    if isinstance(mm, nn.BatchNorm2d):
                        bn_init(mm, 1)
            elif isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        # N, T, V, C 
        x = x.permute(0, 3, 1, 2)[..., None] 

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N C T V M --> N M V C T
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = x.permute(0, 2, 3, 1)

        return x

class Model2(nn.Module):
    def __init__(self, A, num_class, num_point, num_person, block_args, graph_args, kernel_size, block_type, atten,
                 **kwargs):
        super(Model2, self).__init__()

        kwargs['act'] = Activations(kwargs['act'])
        atten = None if atten == 'None' else atten
        
        self.A = A

        self.layers = nn.ModuleList()

        for i, block in enumerate(block_args):
            self.layers.append(MST_GCN_block(in_channels=block[0], out_channels=block[1], residual=block[2],
                                                kernel_size=kernel_size, stride=block[3], A=A, block_type=block_type,
                                                atten=atten, **kwargs))

        for m in self.modules():
            if isinstance(m, SpatialGraphCov) or isinstance(m, Spatial_Sep_Layer):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        conv_branch_init(mm, self.A.shape[0])
                    if isinstance(mm, nn.BatchNorm2d):
                        bn_init(mm, 1)
            elif isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, math.sqrt(2. / num_class))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_args[-1][1], num_class)

    def forward(self, x):
        # N, T, V, C 
        x = x.permute(0, 3, 1, 2)[..., None] 

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).view(N * M, C, T, V)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.gap(x).view(N, M, -1).mean(dim=1)
        x = self.fc(x)

        return x

class MST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, residual, kernel_size, stride, A, block_type, atten, **kwargs):
        super(MST_GCN_block, self).__init__()
        self.atten = atten
        self.msgcn = __block_type__[block_type][0](in_channels=in_channels, out_channels=out_channels, A=A,
                                                   residual=residual, **kwargs)
        self.mstcn = __block_type__[block_type][1](channels=out_channels, kernel_size=kernel_size, stride=stride,
                                                   residual=residual, **kwargs)
        if atten is not None:
            self.att = Attention_Layer(out_channels, atten, **kwargs)

    def forward(self, x):
        return self.att(self.mstcn(self.msgcn(x))) if self.atten is not None else self.mstcn(self.msgcn(x))

