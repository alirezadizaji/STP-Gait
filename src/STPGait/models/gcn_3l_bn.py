from dig.xgraph.models import GCN_3l_BN as GCN
import torch

class GCN_3l_BN(GCN):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, 
            edge_weight: torch.Tensor = None) -> torch.Tensor:
        post_conv = self.relu1(self.conv1(x, edge_index, edge_weight))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index, edge_weight))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out