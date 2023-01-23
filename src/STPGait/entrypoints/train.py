from abc import ABC, abstractmethod
import os
from tqdm import tqdm
from typing import Dict, Generic, List, TYPE_CHECKING, Optional, TypeVar, Union

from dig.xgraph.models import *
import numpy as np
import torch

from ..config import BaseConfig
from .core import MainEntrypoint
from ..enums import Separation

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

T = TypeVar('T', bound=BaseConfig)
IN, OUT = TypeVar('IN'), TypeVar('OUT')
class TrainEntrypoint(MainEntrypoint[T], ABC, Generic[IN, OUT, T]):


    def __init__(self, kfold, conf: T) -> None:
        super().__init__(kfold, conf)

        # criteria defined for train/validation sets to be visualized
        self._criteria_vals = np.full_like((2, self.conf.training_config.num_epochs, len(self.criteria_names)), fill_value=-np.inf)
        self._TRAIN_CRITERION_IDX = 0
        self._VAL_CRITERION_IDX = 1
        self._TEST_CRITERION_IDX = 2

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

    @abstractmethod
    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        """ It calculates loss

        Args:
            x (_type_): model output
            data (_type_): dataloader output

        Returns:
            torch.Tensor: loss value
        """

    @abstractmethod
    def _train_start(self) -> None:
        """ Any tasks that should be accomplished before starting training epoch. """

    @abstractmethod
    def _eval_start(self) -> None:
        """ Any tasks that should be accomplished before starting evaluation epoch. """

    @abstractmethod
    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        """ Any tasks that should be accomplished at the end of the training iteration.

        Args:
            iter_num (int): Iteration number
            x (U): model output
            data (T): dataloader output
        """

    @abstractmethod
    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        """ Any tasks that should be accomplished at the end of the evaluation iteration.

        Args:
            separation (Separation): Loader belongs to this separation
            iter_num (int): Iteration number
            x (U): model output
            data (T): dataloader output
        """

    @abstractmethod
    def _train_epoch_end(self) -> np.ndarray:
        """ Any tasks that should be accomplished at the end of the training epoch. """

    @abstractmethod
    def _eval_epoch_end(self, datasep: Separation) -> np.ndarray:
        """ Any tasks that should be accomplished at the end of the evaluation epoch. """

    @abstractmethod
    def best_epoch_criteria(self, best_epoch: int) -> bool:
        """ Criteria that determines whether best epoch changes or not. """        

    @property
    def criteria_names(self) -> List[str]:
        return ['Loss']

    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('Loss')

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

        criteria = self._train_epoch_end()
        assert criteria.size == len(self.criteria_names), "Mismatch between criteria values length and number of existing names."
        self._criteria_vals[self._TRAIN_CRITERION_IDX, self.epoch] = criteria

    def _validate_for_one_epoch(self, separation: Separation, loader: 'DataLoader') -> Union[None, np.ndarray]:
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

            criteria = self._eval_epoch_end(separation)
            assert criteria.size == len(self.criteria_names), "Mismatch between criteria values length and number of existing names."
            if separation == Separation.VAL:
                self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch] = criteria
            else:
                return criteria

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
            self._validate_for_one_epoch(Separation.VAL, self.val_loader)
            val = self._criteria_vals[1, self.epoch, self.best_epoch_criterion_idx]
            
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
        self.epoch = best_epoch
        self.model.load_state_dict(torch.load(self._get_weight_save_path(self.epoch), map_location=self.conf.device))
        criteria = self._validate_for_one_epoch(Separation.TEST, self.test_loader)
        test = criteria[self.best_epoch_criterion_idx]
        self.fold_test_criterion[self.kfold.testK] = test
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        print(f"@@@@@ Best criteria ValK {self.kfold.valK} epoch {best_epoch} @@@@@\nval: {val}, test: {test}", flush=True)


    def run(self):
        self.fold_test_criterion: Dict[int, float] = dict()

        for _ in range(self.kfold.K):
            with self.kfold:
                self.model = self.get_model()
                self.model.to(self.conf.device)
                self.set_loaders()
                self.set_optimizer(self.conf.training_config.optim_type)
                self._main()
        
        print(f"@@@ Final Result on {self.kfold.K} KFold operation on test set: {np.mean(list(self.fold_test_criterion.values()))} @@@", flush=True)