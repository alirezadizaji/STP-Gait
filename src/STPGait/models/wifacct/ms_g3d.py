from torch import nn
from torch.nn import functional as F

from ..others.msg3d.submodules.ms_gcn import MultiScale_GraphConv as MS_GCN
from ..others.msg3d.submodules.ms_tcn import MultiScale_TemporalConv as MS_TCN
from ..others.msg3d.graph.openpose import AdjMatrixGraph
from ..others.msg3d.model import MultiWindow_MS_G3D

class Model1(nn.Module):
    def __init__(self,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 in_channels=2):
        super(Model1, self).__init__()

        A_binary = AdjMatrixGraph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        c1 = 60
        c2 = c1

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)[..., None] # B, C, T, V, M
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        x = x.permute(0, 2, 3, 1)
        return x

class Model2(nn.Module):
    def __init__(self,
                 num_class,
                 num_gcn_scales,
                 num_g3d_scales):
        super(Model2, self).__init__()

        A_binary = AdjMatrixGraph().A_binary

        # channels
        c1 = 60
        c2 = c1

        self.gcn3d3 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c2, c2)

        self.fc = nn.Sequential(
            nn.Linear(c2, num_class),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # B, C, T, V
        N, *_, M = x.size()

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out