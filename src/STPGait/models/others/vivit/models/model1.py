from torch import nn, Tensor

from .encoder import Encoder

class SpatioTemporalEncoder(Encoder):
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
 