from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn


class FactorizedDotProdAttTransformerEncoderLayer(nn.Module):
    """ Factorized dot product Attention Transformer Encoder Layer proposed in https://arxiv.org/pdf/2103.15691.pdf
    
    NOTE:
        The implementation part is mostly the same as `torch.nn.TransformerEncoderLayer`.
    """
    def __init__(self, q_d, kv_ds, kv_dt, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.Module, self).__init__()

        shead = thead = nhead // 2
        self.self_sattn = nn.MultiheadAttention(q_d, shead, dropout=dropout, kdim=kv_ds, vdim=kv_ds)
        self.self_tattn = nn.MultiheadAttention(q_d, thead, dropout=dropout, kdim=kv_dt, vdim=kv_dt)

        self.sublin = nn.Linear(2 * q_d, q_d)

        self.linear1 = nn.Linear(q_d, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_d)

        self.norm1 = nn.LayerNorm(q_d)
        self.norm2 = nn.LayerNorm(q_d)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(nn.Module, self).__setstate__(state)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): shape B, n_t, n_v, d

        Returns:
            Tensor: _description_
        """
        B, nt, nv, d = x.size()
        assert x.ndim == 4, "X's shape must form (B, n_t, n_v, d)"
        assert d == self.self_sattn.embed_dim == self.self_tattn.embed_dim
        assert nt * d == self.self_tattn.kdim
        assert nv * d == self.self_sattn.kdim
        
        x = x.reshape(B, nt * nv, d)
        xs = x.reshape(B, nt, nv * d)
        xt = x.reshape(B, nv, nt * d)

        x_s = self.self_sattn(x, xs, xs)[0]
        x_t = self.self_tattn(x, xt, xt)[0]
        x2 = torch.cat([x_s, x_t], dim=1)
        
        x = x + self.dropout1(self.sublin(x2))
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x