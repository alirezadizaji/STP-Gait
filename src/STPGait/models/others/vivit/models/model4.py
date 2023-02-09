from torch import nn, Tensor

from .encoder import Encoder
from ..submodules import FactorizedDotProdAttTransformerEncoderLayer as FDPAT

class FactorizedDotProductSelfAttEncoder(Encoder):
    """ Model 4 """
    def __init__(self, q_d, kv_ds, kv_dt, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        enc_layer = FDPAT(q_d, kv_ds, kv_dt, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=None)
        self.q_d = q_d
    
    @property
    def out_dim(self):
        return self.q_d

    def forward(self, x: Tensor) -> Tensor: 
        assert x.ndim == 4, "X's shape must have the form `(B, nt, nv, d)`."
        x = self.encoder(x)
        
        return x