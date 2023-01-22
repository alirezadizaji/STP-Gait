from typing import Tuple

import torch

from .gcn_lstm_transformer import GCNLSTMTransformer

class GCNLSTMTransformerV2(GCNLSTMTransformer):
    def _get_lstm_hidden_state(self, lstm_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hs[lstm_idx] is None:
            self.hs[lstm_idx] = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])
            self.cs[lstm_idx] = self.init_lstm_hidden_state(self.lstm_conf["num_layers"], batch_size, self.lstm_conf["hidden_size"])

        return self.hs[lstm_idx], self.cs[lstm_idx]

    def _update_lstm_hidden_state(self, lstm_idx: int, h: torch.Tensor, c: torch.Tensor) -> None:
        self.hs[lstm_idx] = h
        self.cs[lstm_idx] = c