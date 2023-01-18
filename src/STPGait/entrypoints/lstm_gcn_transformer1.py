import os
from typing import List, Tuple
import pickle

import numpy as np
import torch

from ..config import BaseConfig, TrainingConfig
from ..data.read_gait_data import ProcessingGaitConfig
from ..dataset.KFold import GraphSkeletonKFoldOperator, GraphSkeletonKFoldConfig, KFoldConfig
from ..enums import Optim, Separation
from ..models import GCNLSTMTransformer
from ..preprocess.main import PreprocessingConfig
from .train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class Entrypoint(TrainEntrypoint[IN, OUT, float, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=GraphSkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=10, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=2,
            try_name="lstm_gcn_transformer",
            device="cpu",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super().__init__(kfold, config)

    
    def get_model(self):
        model = GCNLSTMTransformer()
        return model
        
    def _model_forwarding(self, data: IN) -> OUT:
        x, y = data
        x = x[..., [0, 1]].flatten(2)
        x = self.model(x.to(self.conf.device), y.to(self.conf.device))
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        loss1, loss2 = x[1], x[2]
        return 0.2 * loss1 + loss2

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

    def _train_epoch_end(self):
        acc = self.correct / self.total
        print(f'epoch{self.epoch} loss value {np.mean(self.losses)} acc {acc}', flush=True)
    
    def _eval_epoch_end(self, datasep: Separation):
        acc = self.correct / self.total
        print(f'epoch{self.epoch} {datasep} acc {acc}', flush=True)

        print(f'epoch {self.epoch} separation {datasep} loss value {np.mean(self.losses)} acc {acc}', flush=True)
        return np.mean(self.losses)

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self.val_criterias[self.kfold.valK, self.epoch]
        return val <= self.val_criterias[self.kfold.valK, best_epoch]
