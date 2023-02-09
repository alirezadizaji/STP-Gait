from torch import nn, Tensor

from .encoder import Encoder
from ..submodules import FactorizedDotProdAttTransformerEncoderLayer as FDPAT

class FactorizedDotProductSelfAttEncoder(Encoder):
    """ Model 4 """
    def __init__(self, d_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        enc_layer = FDPAT(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=None)
        self.d_model = d_model
    
    @property
    def out_dim(self):
        return self.d_model

    def forward(self, x: Tensor) -> Tensor: 
        assert x.ndim == 4, "X's shape must have the form `(B, nt, nv, d)`."
        x = self.encoder(x)
        
        return x