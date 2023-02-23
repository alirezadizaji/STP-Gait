from dig.xgraph.models import GCNConv, GlobalMeanPool
from torch import nn, Tensor
from torch_geometric.data import Batch, Data

class GCNConv(nn.Module):
    def __init__(self, d1: int, d2: int):
        super().__init__()

        self.conv = GCNConv(d1, d2)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(d2),
            nn.ReLU()
        )

    def forward(self, x: Tensor, edge_index: Tensor):
        B, T, V, _ = x.size()
        x = x.reshape(B, T*V, -1) # B, T*V, C
        data = Batch.from_data_list([Data(x=x_, edge_index=edge_index) for x_ in x])
        
        x = data.x
        x = self.bn_relu(self.conv(x, data.edge_index))
        x = x.reshape(B, T, V, -1)
        return x


class GCNConvFC(GCNConv):
    def __init__(self, d: int, num_classes: int, dropout_p: float = 0.2):
        super().__init__(d, d)
        self.pool = GlobalMeanPool()
        self.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(d, num_classes),
                nn.LogSoftmax(dim=1),
        )
    
    def forward(self, x: Tensor, edge_index: Tensor):
        B, T, V, _ = x.size()
        x = x.reshape(B, T*V, -1) # B, T*V, C
        data = Batch.from_data_list([Data(x=x_, edge_index=edge_index) for x_ in x])
        
        x = data.x
        x = self.bn_relu(self.conv(x, data.edge_index))
        x = self.pool(x, data.batch)
        
        x = self.fc(x)

        return x
