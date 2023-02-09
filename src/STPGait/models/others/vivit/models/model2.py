from torch import nn, Tensor

from .encoder import Encoder


class FactorizedEncoder(Encoder):
    """ Model 2 """
    def __init__(self, d_model, nhead, n_enc_layers, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        s_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.s_encoder = nn.TransformerEncoder(
            encoder_layer=s_enc_layer,
            num_layers=n_enc_layers, 
            norm=None)

        t_enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.t_encoder = nn.TransformerEncoder(
            encoder_layer=t_enc_layer,
            num_layers=n_enc_layers, 
            norm=None)

        self.d_model = d_model

    @property
    def out_dim(self) -> int:
        return self.d_model

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 4, "X's shape must have form of `(B, nt, nv, d)`."
        B, nt, nv, d = x.size()
        x = x.reshape(B * nt, nv, d)

        x = self.s_encoder(x)
        x = x.reshape(B, nt, nv, d)[:, :, 0, :]
        x = self.t_encoder(x)

        return x

