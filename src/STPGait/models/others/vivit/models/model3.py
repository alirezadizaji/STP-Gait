from torch import nn, Tensor

from .encoder import Encoder
from ..submodules import FactorizedSelfAttTransformerEncoderLayer as FSAT


class FactorizedSelfAttEncoder(Encoder):
    """ Model 3 """
    def __init__(self, ds_model, dt_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        enc_layer = FSAT(ds_model, dt_model, nhead, dim_feedforward, dropout, activation)
        self.dt_model = dt_model
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers, norm=None)
    
    @property
    def out_dim(self) -> int:
        return self.dt_model

    def forward(self, x: Tensor) -> Tensor: 
        assert x.ndim == 4, "X's shape must have the form `(B, nt, nv, d)`."
        x = self.encoder(x)

        return x