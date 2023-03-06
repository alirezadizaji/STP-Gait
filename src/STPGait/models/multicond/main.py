from copy import deepcopy
from typing import List, Optional, Tuple, TypeVar, Generic
from typing_extensions import Protocol

import torch
from torch import nn, Tensor
from torchnlp import Attention

T = TypeVar('T', bound=nn.Module)

class AggMode:
    AVG = "Average"
    CAT = "Concatenation"
    ATT = "Attention"

class MultiCond(nn.Module, Generic[T]):
    def __init__(self, basic_block: T, fc_hidden_num: List[int], num_classes: int, 
            agg_mode: AggMode = AggMode.AVG, copy_num: int = 3, z_dim: Optional[int] = None):
        super().__init__()
        
        self.base: nn.ModuleList = nn.ModuleList([])
        for _ in range(copy_num):
            self.base.append(deepcopy(basic_block))

        self.agg_mode = agg_mode
        self.start_dim = fc_hidden_num[0]

        modules = list()
        for h1, h2 in zip(fc_hidden_num[:-1], fc_hidden_num[1:]):
            modules.append(nn.Linear(h1, h2))

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(fc_hidden_num[-1], num_classes))
        modules.append(nn.LogSoftmax(dim=-1))
        self.fc = nn.Sequential(*modules)

        self.small_q = None
        self.attention = None
        if agg_mode == AggMode.ATT:
            assert z_dim is not None, "In Attention mode, z_dim must be given. got None instead."
            self.small_q = nn.Parameter(torch.randn(1, z_dim), requires_grad=True)
            self.attention = Attention(z_dim)
    
    def forward(self, cond_mask: Tensor, inps: List[Tuple[torch.Tensor, ...]]) -> Tensor:
        out = []
        for inp, m in zip(inps, self.base):
            out.append(m(*inp).flatten(1))
        
        x = torch.stack(out, dim=1)                             # B, M, Z
        B = x.size(0)
        assert torch.all(cond_mask), "All aggregation modes are currently available only for samples whose all conditions are valid."
        if self.agg_mode == AggMode.AVG:
            x = x.mean(1)
        elif self.agg_mode == AggMode.CAT:
            x = x.flatten(1)
        else:
            big_q = torch.tile(self.small_q, (B, 1)).unsqueeze(1)
            x = self.attention(big_q, x).squeeze(1)

        assert x.ndim == 2 and x.size(1) == self.start_dim, \
                """ Number of X dimension must be two and the last one must be matched with first layer FC. """
        
        x = self.fc(x)
        return x