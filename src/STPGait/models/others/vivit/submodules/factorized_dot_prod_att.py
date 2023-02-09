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
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        shead = thead = nhead // 2
        self.self_sattn = nn.MultiheadAttention(d_model, shead, dropout=dropout)
        self.self_tattn = nn.MultiheadAttention(d_model, thead, dropout=dropout)

        self.sublin = nn.Linear(2 * d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        """
        Args:
            x (Tensor): shape B, n_t, n_v, d

        Returns:
            Tensor: _description_
        """
        B, nt, nv, d = x.size()
        assert x.ndim == 4, "X's shape must form (B, n_t, n_v, d)"
        assert d == self.self_sattn.embed_dim == self.self_tattn.embed_dim
        
        x_3d = x.reshape(B, nt * nv, d)
        xs = x.reshape(B * nt, nv, d)
        xt = x.transpose(1, 2).reshape(B * nv, nt, d)

        x_s = self.self_sattn(xs, xs, xs)[0].reshape(B, nt * nv, d)
        x_t = self.self_tattn(xt, xt, xt)[0].reshape(B, nv, nt, d).transpose(1, 2).reshape(B, nt * nv, d)
        x2 = torch.cat([x_s, x_t], dim=2)
        
        x2 = self.dropout1(self.sublin(x2))
        x2 = self.norm1(x2)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x2 = x_3d + self.dropout2(x2)
        x2 = self.norm2(x2)
        x2 = x2.reshape(B, nt, nv, d)
        return x2