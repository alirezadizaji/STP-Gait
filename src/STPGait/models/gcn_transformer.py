from typing import List, Optional

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .gcn_lstm_transformer import TransformerEncoderConf
from .gcn_3l_bn import GCN_3l_BN

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

    def forward(self, batch_size: int, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, 
            edge_weight: torch.Tensor = None) -> torch.Tensor:
        N = batch_size
        L = x.size(0)
        T = L // (N * self.V)

        post_conv = self.relu1(self.conv1(x, edge_index, edge_weight))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index, edge_weight))
        
        post_conv = post_conv.reshape(N, T, self.V, -1)
        assert post_conv.size(1) == T and post_conv.size(2) == self.V, "Shape mismatch."
        
        post_conv = post_conv.reshape(N, T, -1)  # N, T, V*D
        post_conv = self.encoder(post_conv)
        post_conv = post_conv.reshape(N, T, self.V, -1).reshape(N*T*self.V, -1)

        out_readout = self.readout(post_conv, batch)
        out = self.ffn(out_readout)

        return out