""" ViViT Model implementation, paper: https://arxiv.org/pdf/2103.15691.pdf """


from torch import nn
from torch import nn, Tensor

from .submodules import FactorizedDotProdAttTransformerEncoderLayer, FactorizedSelfAttTransformerEncoderLayer

class _SpatioTemporalEncoder(nn.Module):
    """ Model 1 """
    def __init__(self, d_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.Module, self).__init__()

        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer,
            num_layers=n_enc_layers, norm=None)
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "X's shape must have form of `(B, nt, nv, d)`."
        B, nt, nv, d = x.size()
        x = x.reshape(B, nt * nv, d)
        x = self.encoder(x)

        return x
 
class _FactorizedEncoder(nn.Module):
    """ Model 2 """
    def __init__(self, ds_model, dt_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.Module, self).__init__()
        
        s_enc_layer = nn.TransformerEncoderLayer(ds_model, nhead, dim_feedforward, dropout, activation)
        self.s_encoder = nn.TransformerEncoder(
            encoder_layer=s_enc_layer,
            num_layers=n_enc_layers, 
            norm=None)

        t_enc_layer = nn.TransformerEncoderLayer(dt_model, nhead, dim_feedforward, dropout, activation)
        self.t_encoder = nn.TransformerEncoder(
            encoder_layer=t_enc_layer,
            num_layers=n_enc_layers, 
            norm=None)
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "X's shape must have form of `(B, nt, nv, d)`."
        B, nt, nv, d = x.size()
        x = x.reshape(B * nt, nv, d)

        x = self.s_encoder(x)
        x = x.reshape(B, nt, nv, d)[:, :, 0, :]
        x = self.t_encoder(x)

        return x


class _FactorizedSelfAttEncoder(nn.Module):
    """ Model 3 """
    def __init__(self, ds_model, dt_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.Module, self).__init__()

        enc_layer = FactorizedSelfAttTransformerEncoderLayer(ds_model, dt_model, nhead, dim_feedforward, dropout, activation)
        
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=None)
    
    def forward(self, x: Tensor) -> Tensor: 
        assert x.ndim == 4, "X's shape must have the form `(B, nt, nv, d)`."
        x = self.encoder(x)

        return x

class _FactorizedDotProductSelfAttEncoder(nn.Module):
    """ Model 4 """
    def __init__(self, q_d, kv_ds, kv_dt, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(nn.Module, self).__init__()

        enc_layer = FactorizedDotProdAttTransformerEncoderLayer(q_d, kv_ds, kv_dt, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=None)
    
    def forward(self, x: Tensor) -> Tensor: 
        assert x.ndim == 4, "X's shape must have the form `(B, nt, nv, d)`."
        x = self.encoder(x)
        
        return x

class ViViT(nn.Module):
    def __init__(self):
        pass