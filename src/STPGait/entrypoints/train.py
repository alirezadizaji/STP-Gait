from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from typing import Dict, Generic, TYPE_CHECKING, Optional, TypeVar

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F

from ..config import BaseConfig
from .core import MainEntrypoint
from ..enums import Separation, Optim

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

T = TypeVar('T', bound=BaseConfig)
IN, OUT, C = TypeVar('IN'), TypeVar('OUT'), TypeVar('C')
class TrainEntrypoint(MainEntrypoint[T], ABC, Generic[IN, OUT, C, T]):

    def _get_weight_save_path(self, epoch: int) -> str:
        weight_save_path =  os.path.join(self.conf.save_dir, "weights", f"{self.kfold.valK}-{self.kfold.testK}", str(epoch))
        os.makedirs(os.path.dirname(weight_save_path), exist_ok=True)
        return weight_save_path

    @abstractmethod
    def _model_forwarding(self, data: IN) -> OUT:
        """ forwards the model, using dataloader output

        Args:
            data (_type_): dataloader output

        Returns:
            (_type_): model output
        """
        # x = self.model(data)
        # return x

    @abstractmethod
    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        """ It calculates loss

        Args:
            x (_type_): model output
            data (_type_): dataloader output

        Returns:
            torch.Tensor: loss value
        """
        # x = F.log_softmax(x, dim=-1)
        # loss = - torch.mean(x[torch.arange(x.size(0)), data])
        # return loss

    @abstractmethod
    def _train_start(self) -> None:
        """ Any tasks that should be accomplished before starting training epoch. """
        # self.correct = self.total = 0
        # self.losses = list()

    @abstractmethod
    def _eval_start(self) -> None:
        """ Any tasks that should be accomplished before starting evaluation epoch. """
        # self._train_start()

    @abstractmethod
    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        """ Any tasks that should be accomplished at the end of the training iteration.

        Args:
            iter_num (int): Iteration number
            x (U): model output
            data (T): dataloader output
        """
        # self.losses.append(loss.item())

        # y_pred = x.argmax(-1)
        # self.correct += torch.sum(y_pred == data.y).item()
        # self.total += data.y.numel()

        # if iter_num % 20 == 0:
        #     print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    @abstractmethod
    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        """ Any tasks that should be accomplished at the end of the evaluation iteration.

        Args:
            separation (Separation): Loader belongs to this separation
            iter_num (int): Iteration number
            x (U): model output
            data (T): dataloader output
        """
        # y_pred = x.argmax(-1)
        # self.correct += torch.sum(y_pred == data.y).item()
        # self.total += data.y.numel()

    @abstractmethod
    def _train_epoch_end(self) -> None:
        """ Any tasks that should be accomplished at the end of the training epoch. """
        # print(f'epoch {self.epoch} train acc {self.correct/self.total}', flush=True)

    @abstractmethod
    def _eval_epoch_end(self, datasep: Separation) -> C:
        """ Any tasks that should be accomplished at the end of the evaluation epoch. """
        # acc = self.correct / self.total
        # print(f'epoch{self.epoch} {datasep} acc {acc}', flush=True)

    @abstractmethod
    def best_epoch_criteria(self, best_epoch: int) -> bool:
        """ Criteria that determines whether best epoch changes or not. """        
        # val = self.val_criterias[self.kfold.valK, self.epoch]
        # return val > self.val_criterias[self.kfold.valK, best_epoch]

    def _save_model_weight(self) -> None:
        torch.save(self.model.state_dict(), self._get_weight_save_path(self.epoch))

    def _remove_model_weight(self, best_epoch: int) -> None:
        os.remove(self._get_weight_save_path(best_epoch))

    def _train_for_one_epoch(self) -> None:
        self.model.train()

        self._train_start()
        for i, data in enumerate(self.train_loader):
            # Switch data device 
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].to(self.conf.device)

            x: OUT = self._model_forwarding(data)
            loss = self._calc_loss(x, data)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self._train_iter_end(i, loss, x, data)

        self._train_epoch_end()

    def _validate_for_one_epoch(self, separation: Separation, loader: 'DataLoader') -> C:
        self.model.eval()

        self._eval_start()
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Switch data device 
                for i in range(len(data)):
                    if isinstance(data[i], torch.Tensor):
                        data[i] = data[i].to(self.conf.device)

                x = self._model_forwarding(data)
                loss = self._calc_loss(x, data)

                self._eval_iter_end(i, separation, loss, x, data)

            return self._eval_epoch_end(separation)            

    def _early_stopping(self, best_epoch: Optional[int], thd: int = 50) -> bool:
        best = best_epoch if best_epoch is not None else 0
        difference = self.epoch - best
        if difference >= thd:
            return True
        
        return False
    
    def _main(self):
        num_epochs = self.conf.training_config.num_epochs
        best_epoch = None
        for self.epoch in tqdm(range(num_epochs)):
            self._train_for_one_epoch()
            val: C = self._validate_for_one_epoch(Separation.VAL, self.val_loader) 
            self.val_criterias[self.kfold.valK, self.epoch] = val
            
            # save only best epoch in terms of validation accuracy             
            if best_epoch is None or self.best_epoch_criteria(best_epoch):
                if best_epoch is not None:
                    self._remove_model_weight(best_epoch)
                self._save_model_weight()
                best_epoch = self.epoch
                print(f"### Best epoch changed to {best_epoch} criteria {val} ###", flush=True)

            # check early stopping if necessary
            if self.conf.training_config.early_stop is not None \
                    and self._early_stopping(best_epoch, self.conf.training_config.early_stop):
                print(f"*** EARLY STOPPING: epoch No. {self.epoch}, best epoch No. {best_epoch} ***", flush=True)
                break

        # evaluate best epoch on test set
        self.model.load_state_dict(torch.load(self._get_weight_save_path(best_epoch), map_location=self.conf.device))
        test: C =self._validate_for_one_epoch(Separation.TEST, self.test_loader)
        self.test_criterias[self.kfold.testK] = test
        val: C = self.val_criterias[self.kfold.valK, best_epoch]
        print(f"@@@@@ Best criteria ValK {self.kfold.valK} epoch {best_epoch} @@@@@\nval: {val}, test: {test}", flush=True)


    def run(self):
        self.val_criterias: Dict[int, Dict[int, C]] = dict()
        self.test_criterias: Dict[int, C] = dict()

        for _ in range(self.kfold.K):
            with self.kfold:
                self.model = self.get_model()
                self.model.to(self.conf.device)
                self.set_loaders()
                self.set_optimizer(self.conf.training_config.optim_type)
                self._main()
        
        print(f"@@@ Final Result on {self.kfold.K} KFold operation on test set: {np.mean(list(self.test_criterias.values()))} @@@", flush=True)