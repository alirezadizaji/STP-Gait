from typing import List, Optional

from dig.xgraph.models import GCN_3l_BN
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .gcn_lstm_transformer import TransformerEncoderConf

class GCNTransformer(GCN_3l_BN):
    def __init__(self, model_level: str, dim_node: int, 
            dim_hidden: int, 
            num_classes: int, 
            num_nodes: int,
            transformer_encoder_conf: Optional[TransformerEncoderConf] = TransformerEncoderConf()):
        super().__init__(model_level, dim_node, dim_hidden, num_classes)

        transformer_encoder_conf = transformer_encoder_conf.__dict__
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=num_nodes * dim_hidden,
                nhead=transformer_encoder_conf['enc_n_heads'])        
        
        self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=transformer_encoder_conf['n_enc_layers'], 
                norm=None)

        self.V = num_nodes

    def forward(self, data: List[Data]) -> torch.Tensor:
        N = len(data)
        L, _ = data[0].x.shape
        T = L // self.V

        with torch.no_grad():
            X: Batch = Batch.from_data_list(data)
            x, edge_index, batch = X.x, X.edge_index, X.batch

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        
        post_conv = post_conv.reshape(N, T, self.V, -1)
        assert post_conv.size(1) == T and post_conv.size(2) == self.V, "Shape mismatch."
        
        post_conv = post_conv.reshape(N, T, -1)  # N, T, V*D
        post_conv = self.encoder(post_conv)
        post_conv = post_conv.reshape(N, T, self.V, -1).reshape(N*T*self.V, -1)

        out_readout = self.readout(post_conv, batch)
        out = self.ffn(out_readout)

        return out