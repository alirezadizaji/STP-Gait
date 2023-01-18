from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .skeleton import SkeletonDataset

class GraphSkeletonDataset(SkeletonDataset):

    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]