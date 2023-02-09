from typing import Optional

from torch import nn, Tensor


class FactorizedSelfAttTransformerEncoderLayer(nn.Module):
    """ Factorized Self Attention Transformer Encoder Layer proposed in https://arxiv.org/pdf/2103.15691.pdf """
    def __init__(self, ds_model, dt_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.spatial_enc = nn.TransformerEncoderLayer(ds_model, nhead, dim_feedforward, dropout, activation)
        self.temporal_enc = nn.TransformerEncoderLayer(dt_model, nhead, dim_feedforward, dropout, activation)
    
    def forward(self, x: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        """
        Args:
            x (Tensor): shape B, n_t, n_v, d

        Returns:
            Tensor: _description_
        """

        B, nt, nv, d = x.size()
        assert x.ndim == 4, "X's shape must form (B, n_t, n_v, d)"
        assert nv * d == self.spatial_enc.self_attn.embed_dim
        assert nt * d == self.temporal_enc.self_attn.embed_dim
        
        x = x.reshape(B, nt, nv * d)
        x = self.spatial_enc(x)
        
        x = x.reshape(B, nt, nv, d).transpose(1, 2).reshape(B, nv, nt * d)
        x = self.temporal_enc(x).transpose(1, 2).reshape(B, nt, nv, d)

        return x