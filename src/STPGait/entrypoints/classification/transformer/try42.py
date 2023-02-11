from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ....config import BaseConfig, TrainingConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Separation
from ....models import Transformer
from ....preprocess.main import PreprocessingConfig
from ..lstm_gcn_transformer.try4 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 42 (try 4 ->)
## Use transformer model.
## Filter out hardcases
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0),
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                filterout_hardcases=True,
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=42,
            try_name="transformer",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)


    def get_model(self):
        return Transformer()
    
    def _model_forwarding(self, data: IN) -> OUT:
        x, y, *_ = data
        x = x[..., [0, 1]].flatten(2)
        x = self.model(x.to(self.conf.device), y.to(self.conf.device))
        return x

    def set_loaders(self) -> None:
        self.train_loader = DataLoader(self.kfold.train, batch_size=self.conf.training_config.batch_size, shuffle=self.conf.training_config.shuffle_training, drop_last=True)
        self.val_loader = DataLoader(self.kfold.val, batch_size=self.conf.eval_batch_size, drop_last=True)
        self.test_loader = DataLoader(self.kfold.test, batch_size=self.conf.eval_batch_size, drop_last=True)

    @property
    def criteria_names(self) -> List[str]:
        return ['LOSS', 'ACC']
    
    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')
       
    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        loss = x[1]
        self.losses.append(loss.item())

        return loss

    def _train_start(self) -> None:
        self.correct = self.total = 0
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        x_probs = x[0]
        y_pred = x_probs.argmax(-1)
        y = data[1]
        self.correct += torch.sum(y_pred == y).item()
        self.total += y.numel()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, iter_num: int, separation: Separation, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())
        
        x_probs = x[0]
        y_pred = x_probs.argmax(-1)
        y = data[1]
        self.correct += torch.sum(y_pred == y).item()
        self.total += y.numel()

    def _train_epoch_end(self) -> np.ndarray:
        loss = np.mean(self.losses)
        acc = self.correct / self.total
        print(f'epoch{self.epoch} loss value {loss} acc {acc}', flush=True)

        return np.array([loss, acc])
    
    def _eval_epoch_end(self, datasep: Separation) -> np.ndarray:
        acc = self.correct / self.total
        loss = np.mean(self.losses)
        print(f'epoch {self.epoch} separation {datasep} loss value {loss} acc {acc}', flush=True)
        return np.array([loss, acc])

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        best = self._criteria_vals[self._VAL_CRITERION_IDX, best_epoch, self.best_epoch_criterion_idx]
        return val > best
