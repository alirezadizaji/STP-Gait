from copy import deepcopy
from typing import Dict, Generic, List, Optional, TypeVar

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch, Data

M1, M2 = TypeVar('M1'), TypeVar('M2')
class WiFaCCT(nn.Module, Generic[M1, M2]):
    """ WiFaCCT: Window Framing Cross Consistency Training. """
    
    def __init__(self, model1: M1, model2: M2, num_frames: Optional[int] = None, num_aux_branches: int = 8) -> None:
        super().__init__()
        
        self.shared = model1
        self.main = model2
        self.auxs: nn.ModuleList = nn.ModuleList(
            [deepcopy(model2) for _ in range(num_aux_branches)])

        self.num_splits: int = num_aux_branches + 1
        self.num_frames: Optional[int] = num_frames

    def forward(self, x: Tensor, m1_args: Dict[str, Tensor], m2_args: Dict[str, Tensor]) -> Tensor:
        B, T, V, _ = x.size()
        if self.num_frames is None:
            self.num_frames = T

        x = self.shared(x, **m1_args)
        assert x.size(0) == B and x.size(1) == self.num_frames and x.size(2) == V, f"Shape mismatch (got {x.size()}, expected {B}, {self.num_frames}, {V}, _)."
        
        o_main = self.main(x, **m2_args)        # B, C

        window_size = int(self.num_frames / self.num_splits)
        si = window_size * (1 + torch.arange(self.num_splits - 1))
        fi = torch.arange(self.num_frames, dtype=torch.int64).repeat(2)

        o_auxs: List[torch.Tensor] = list()
        for i in range(self.num_splits - 1):
            idx = fi[si[i]: si[i] + self.num_frames]

            o_aux = self.auxs[i](x[:, idx], **m2_args)
            o_auxs.append(o_aux)

        o_aux = torch.stack(o_auxs, dim=1)              # B, A, C

        return o_main, o_aux