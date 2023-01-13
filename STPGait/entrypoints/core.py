from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING, Tuple

from torch.optim import Adam, SGD
from torch import nn
from torch.utils.data import DataLoader

from ..enums import *
from ..dataset.KFold import KFoldInitializer, KFoldSkeleton

if TYPE_CHECKING:
    from torch.optim import Optimizer

class MainEntrypoint(ABC):
    def __init__(self, kfold: KFoldInitializer, model: nn.Module) -> None:
        self.model = model
        self.model.to("cuda:0")

        self.kfold: KFoldInitializer = kfold
        self.optimizer: 'Optimizer' = self.get_optimizer()
        self.train_loader = self.val_loader = self.test_loader = None

    def set_loaders(self) -> None:
        self.train_loader = DataLoader(self.kfold.train, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.kfold.val, batch_size=1)
        self.test_loader = DataLoader(self.kfold.test, batch_size=1)

    def get_optimizer(self, optim_type: Optim) -> 'Optimizer':
        if optim_type == Optim.ADAM:
            return Adam(self.model.parameters(), 1e-3)
        elif optim_type == Optim.SGD:
            return SGD(self.model.parameters(), 1e-3)
            
    @abstractmethod
    def run(self):
        """ Define running of the entrypoint here """