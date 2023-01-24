from abc import ABC, abstractmethod
import os
from typing import Generic, TypeVar, TYPE_CHECKING

from torch.optim import Adam, SGD
from torch import nn
from torch.utils.data import DataLoader

from ..config import BaseConfig
from ..enums import *
from ..dataset.KFold import KFoldOperator

if TYPE_CHECKING:
    from torch.optim import Optimizer

T = TypeVar('T', bound=BaseConfig)
class MainEntrypoint(ABC, Generic[T]):
    def __init__(self, kfold: KFoldOperator, conf: T) -> None:
        self.conf: T = conf
    
        self.kfold: KFoldOperator = kfold
        self.set_optimizer(self.conf.training_config.optim_type)
        self.train_loader = self.val_loader = self.test_loader = None

        self.model: nn.Module = self.get_model()
        self.model.to(self.conf.device)
    
    def set_loaders(self) -> None:
        self.train_loader = DataLoader(self.kfold.train, batch_size=self.conf.training_config.batch_size, shuffle=self.conf.training_config.shuffle_training)
        self.val_loader = DataLoader(self.kfold.val, batch_size=self.conf.eval_batch_size)
        self.test_loader = DataLoader(self.kfold.test, batch_size=self.conf.eval_batch_size)

    def set_optimizer(self, optim_type: Optim) -> 'Optimizer':
        if optim_type == Optim.ADAM:
            self.optimizer: 'Optimizer' = Adam(self.model.parameters(), self.conf.training_config.lr)
        elif optim_type == Optim.SGD:
            self.optimizer: 'Optimizer' = SGD(self.model.parameters(), self.conf.training_config.lr)
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """ It returns the model """

    @abstractmethod
    def run(self):
        """ Define running of the entrypoint here """