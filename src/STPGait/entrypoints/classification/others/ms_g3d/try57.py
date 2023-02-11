from typing import List, Tuple

import numpy as np
import torch

from .....config import BaseConfig, TrainingConfig
from .....data.read_gait_data import ProcessingGaitConfig
from .....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from .....enums import Optim, Separation
from .....models.others.msg3d.model import Model
from .....preprocess.main import PreprocessingConfig
from ....train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# Try 57
# Model: MS-G3D
# KFOLD = 5, Validation and Test are the same
# Filterout hardcases
# Filterout unlabeled cases
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                filterout_hardcases=True,
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=57,
            try_name="ms_g3d",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC']
    
    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')
       
    def get_model(self):
        num_classes = self.kfold._ulabels.size
        model = Model(
            num_class=num_classes,
            num_point=25,
            num_person=1,
            num_gcn_scales=7,
            num_g3d_scales=4)
        return model
        
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0]     # B, T, V, C
        x = x.permute(0, 3, 1, 2)[..., None] # B, C, T, V, M
        x = self.model(x)
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        y = data[1]
        idx = torch.arange(y.numel())
        loss = -torch.mean(x[idx, y]) #CE
        return loss

    def _train_start(self) -> None:
        self.correct = self.total = 0
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        x_probs = x
        y_pred = x_probs.argmax(-1)
        y = data[1]
        self.correct += torch.sum(y_pred == y).item()
        self.total += y.numel()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, iter_num: int, separation: Separation, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())
        
        x_probs = x
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
