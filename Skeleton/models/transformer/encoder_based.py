from typing import Tuple

import torch
from torch import nn

class Transformer(nn.Module):
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
    def __init__(self, inp_dim: int = 50, d_model: int = 256, enc_n_heads: int = 6,
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, T, _, _ = x.size()
        x = x.reshape(N, T, -1)                             # N, T, V, C -> N, T, L
        y = x.clone()

        mask = torch.zeros(N, T).bool().to(x.device)
        if self.training:
            idx = torch.randint(0, T, size=(N, int(self._mask_ratio * T)))
            mask[idx] = True
        else:
            # For evaluation, mask only particular frames
            mask[:, 1::10] = True
        mask = mask.unsqueeze(2)

        x[mask] = self._mask_fill_value
        x = self.encoder_lin(x)
        x = self.encoder(x)
        x = self.revert_lin(x)

        if self._apply_loss_in_mask_loc:
            loss = self._mse(x[mask], y[mask])
        else:
            loss = self._mse(x, y)
        
        return x, loss, mask
        