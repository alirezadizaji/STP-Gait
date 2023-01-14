from typing import Tuple

import torch
from torch import nn

class SimpleTransformer(nn.Module):
    r""" This Transformer does not have decoder and instead before and after the encoder 
    part, there are two linear modules to map input into a dimension space and then 
    revert it back to original space.

    Args:
        inp_dim (int, optional): Input dimension at the beginning of forward. Defaults to `50`.
        d_model (int, optional): Encoder layer dimension size. Defaults to `512`.
        enc_n_heads (int, optional): Encoder layer number of heads. Defaults to `6`.
        n_enc_layers (int, optional): Encoder number of layers. Defaults to `3`.
        mask_ratio (float, optional): Input area ratio for masking and then rebuilding. Defaults to `0.01`.
        mask_fill_value (float, optional): Value to fill masked locations. Defaults to `-1e3`.
        apply_loss_in_mask_loc (bool, optional): If `True`, then apply mean squared loss only to masked location, O.W. on the whole part. Defaults to `False`.
    """
    def __init__(self, inp_dim: int = 50, d_model: int = 256, enc_n_heads: int = 8,
            n_enc_layers: int = 3, mask_ratio: float = 0.01, mask_fill_value: float = -1e3,
            apply_loss_in_mask_loc: bool = False) -> None:
        super().__init__()
        
        self._mask_ratio: float = mask_ratio
        self._mask_fill_value: float = mask_fill_value
        self._apply_loss_in_mask_loc: bool = apply_loss_in_mask_loc

        self.encoder_lin = nn.Linear(inp_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=enc_n_heads, 
            batch_first=True)        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_enc_layers, 
            norm=None)
        
        self.revert_lin = nn.Linear(d_model, inp_dim)

        self._mse = nn.MSELoss()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = x.clone()

        x = self.encoder_lin(x)
        x = self.encoder(x)
        x = self.revert_lin(x)

        if self.training:
            if self._apply_loss_in_mask_loc:
                loss = self._mse(x[mask], y[mask])
            else:
                loss = self._mse(x, y)
        else:
            loss = self._mse(x[~mask], y[~mask])
                    
        return x, loss, mask
        