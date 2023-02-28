from copy import deepcopy
from typing import List, Tuple, TypeVar, Generic
from typing_extensions import Protocol

import torch
from torch import nn, Tensor

T = TypeVar('T', bound=nn.Module)

class AvgOp(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...

class MultiCond(nn.Module, Generic[T]):
    def __init__(self, basic_block: T, fc_hidden_num: List[int], num_classes: int, avg_op: AvgOp, copy_num: int = 3):
        super().__init__()
        
        self.base: nn.ModuleList = nn.ModuleList([])
        for _ in range(copy_num):
            self.base.append(deepcopy(basic_block))

        self.avg_op = avg_op
        self.start_dim = fc_hidden_num[0]

        modules = list()
        for h1, h2 in zip(fc_hidden_num[:-1], fc_hidden_num[1:]):
            modules.append(nn.Linear(h1, h2))

        modules.append(nn.Dropout(0.2))
        modules.append(nn.Linear(fc_hidden_num[-1], num_classes))
        modules.append(nn.LogSoftmax(dim=-1))
        self.fc = nn.Sequential(*modules)
    
    def forward(self, cond_mask: Tensor, inps: List[Tuple[torch.Tensor, ...]]) -> Tensor:
        out = []
        for inp, m in zip(inps, self.base):
            out.append(m(*inp))
        
        x = torch.stack(out, dim=0)
        x_interface = torch.zeros_like(x, requires_grad=False)
        x_interface[cond_mask] = x_interface[cond_mask] + x[cond_mask]
        x = x_interface.sum(0)
        x = self.avg_op(x)
        assert x.ndim == 2 and x.size(1) == self.start_dim, \
                """ Number of X dimension must be two and the last one must be matched with first layer FC. """
        x = self.fc(x)
        return x