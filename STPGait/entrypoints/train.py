from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from typing import Generic, TYPE_CHECKING, Optional, TypeVar

from dig.xgraph.models import *
import numpy as np
import torch
from torch.nn import functional as F

from .core import MainEntrypoint
from ..enums import Separation

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

T, U = TypeVar('T'), TypeVar('U')
class TrainEntrypoint(MainEntrypoint, ABC, Generic[T], Generic[U]):

    def _get_weight_save_path(self):
        weight_dir_name = 'weights'
        weight_save_path =  os.path.join("../Results/1_transformer", weight_dir_name, str(self.epoch), f"{self.kfold.valK}-{self.kfold.testK}")
        os.makedirs(weight_save_path, exist_ok=True)
        return weight_save_path

    @abstractmethod
    def _model_forwarding(self, data: T):
        """ forwards the model, using dataloader output

        Args:
            data (_type_): dataloader output

        Returns:
            (_type_): model output
        """
        # x = self.model(data)
        # return x

    @abstractmethod
    def _calc_loss(self, x: U, data: T) -> torch.Tensor:
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
    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: U, data: T) -> None:
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
    def _eval_iter_end(self, iter_num: int, loss: torch.Tensor, x: U, data: T) -> None:
        """ Any tasks that should be accomplished at the end of the evaluation iteration.

        Args:
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
    def _eval_epoch_end(self, datasep: Separation) -> None:
        """ Any tasks that should be accomplished at the end of the evaluation epoch. """
        # acc = self.correct / self.total
        # print(f'epoch{self.epoch} {datasep} acc {acc}', flush=True)

    def _save_model_weight(self) -> None:
        torch.save(self.model.state_dict(), self._get_weight_save_path())

    def _remove_model_weight(self) -> None:
        os.remove(self._get_weight_save_path(self.epoch))

    def _train_for_one_epoch(self) -> None:
        self.model.train()

        self._train_start()
        for i, data in enumerate(self.train_loader):
            data: T = data.to("cuda:0")

            x: U = self._model_forwarding(data)
            loss = self._calc_loss(x, data)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self._train_iter_end(i, loss, x, data)

        self._train_epoch_end()

    def _validate_for_one_epoch(self, loader: 'DataLoader') -> None:
        self.model.eval()

        self._eval_start()
        with torch.no_grad():
            for i, data in enumerate(loader):
                data: T = data.to("cuda:0")
                x = self._model_forwarding(data)
                self._eval_iter_end(i, None, x, data)

            self._eval_epoch_end()            
            return acc

    def _early_stopping(self, epoch: int, best_epoch: Optional[int]) -> bool:
        thd = 50
        best = best_epoch if best_epoch is not None else 0
        difference = epoch - best
        if difference >= thd:
            return True
        
        return False
        
    def run(self):
        num_epochs = 200
        val_accs = np.zeros(num_epochs)
        best_epoch = None
        for self.epoch in tqdm(range(num_epochs)):
            self._train_for_one_epoch()
            val = self._validate_for_one_epoch(self.val_loader) 
            val_accs[self.epoch] = val

            # save only best epoch in terms of validation accuracy             
            if best_epoch is None or val > val_accs[best_epoch]:
                if best_epoch is not None:
                    self._remove_model_weight(best_epoch)
                self._save_model_weight()
                best_epoch = self.epoch
                print(f"### Best epoch changed to {best_epoch} acc {val} ###", flush=True)

            # check early stopping if necessary
            if self._early_stopping(best_epoch):
                print(f"*** EARLY STOPPING: epoch No. {self.epoch}, best epoch No. {best_epoch} ***", flush=True)
                break

        # evaluate best epoch on test set
        best_epoch = val_accs.argmax()
        self.model.load_state_dict(torch.load(self._get_weight_save_path(best_epoch), map_location="cuda:0"))
        test_acc =self._validate_for_one_epoch(self.test_loader)
        val_acc = val_accs[best_epoch]
        print(f"@@@@@ Best accuracy epoch {best_epoch} @@@@@\nval: {val_acc}, test: {test_acc}", flush=True)


    def last_task(self):
        pass