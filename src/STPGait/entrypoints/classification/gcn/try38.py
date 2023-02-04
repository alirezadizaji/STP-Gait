from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Body, Separation, Optim
from ....models import GCNSemiSupervised
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# try 38
## Applying inter-frame edge connection using mode 2 with chunk size 60 and stride 30
## Network is an encoder with three GCN3l networks, first one used for supervised learning, the other two for unsupervised training.
## KFold validation with K = 5 without test set.
## Unlabeled cases are obviously not filtered.
## remove hard cases
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=False),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=38,
            try_name="triple_gcn3l_m2_I_60_offset_30",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
        self._edge_index_upper: torch.Tensor = None
        self._edge_index_lower: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = np.unique(self.kfold.get_labels()).size - 1
        print(f"*** Train GCNSemiSupervised on {num_classes} classes. ***", flush=True)
        model = GCNSemiSupervised(dim_node=2, dim_hidden=60, sup_num_classes=num_classes, unsup_num_classes=num_classes)
        return model

    def _get_edges(self, num_frames: int, body_part: Body) -> torch.Tensor:
        return Skeleton.get_interframe_edges_mode2(num_frames, I=60, offset=30, body_part=body_part)

    def _model_forwarding(self, data: IN) -> OUT:
        x, y, node_invalid, labeled = data
        x = x[..., [0, 1]]
        node_invalid = node_invalid.to(x.device)
        
        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1), Body.WHOLE).to(x.device)
            self._edge_index_upper = self._get_edges(x.size(1), Body.UPPER).to(x.device)
            self._edge_index_lower = self._get_edges(x.size(1), Body.LOWER).to(x.device)

        out: OUT = self.model(x, self._edge_index, self._edge_index_lower, 
                        self._edge_index_upper, y, node_invalid, labeled)
        return out

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        *a, sup_loss, unsup_lower, unsup_upper = x

        loss = 0
        if sup_loss is not None:
            self.sup_losses.append(sup_loss.item())
            loss = loss + sup_loss
            
        loss = loss + 0.2 * (unsup_lower + unsup_upper) 
        self.unsups_lower.append(unsup_lower.item())
        self.unsups_upper.append(unsup_upper.item())
        self.losses.append(loss.item())
        
        return loss

    def _train_start(self) -> None:
        self.correct_sup = self.total_sup = 0
        self.correct_all = self.total_all = 0
        self.losses = list()
        self.sup_losses = list()
        self.unsups_lower = list()
        self.unsups_upper = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        y, labeled = data[1], data[3]
        y_pred, yl, yu, *b = x
    
        if y_pred is not None:
            x_probs = x[0]
            y_pred = x_probs.argmax(-1)
            y = data[1]
            self.correct_sup += torch.sum(y_pred == y[labeled]).item()
            self.total_sup += y.numel()

        self.correct_all += torch.sum(yl == yu).item()
        self.total_all += yl.numel()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        y, labeled = data[1], data[3]
        y_pred, yl, yu, *b = x

        if y_pred is not None:
            x_probs = x[0]
            y_pred = x_probs.argmax(-1)
            y = data[1]
            self.correct_sup += torch.sum(y_pred == y[labeled]).item()
            self.total_sup += y.numel()

        self.correct_all += torch.sum(yl == yu).item()
        self.total_all += yl.numel()


    def _train_epoch_end(self) -> np.ndarray:
        loss = np.mean(self.losses)
        sloss = np.mean(self.sup_losses)
        ulloss = np.mean(self.unsups_lower)
        uuloss = np.mean(self.unsups_upper)

        acc = self.correct_sup / self.total_sup
        uacc = self.correct_all / self.total_all
        print(f'epoch{self.epoch} loss value {loss}, sloss {sloss}, ulloss {ulloss}, uuloss {uuloss}; acc supervised {acc}, unsupervised {uacc}', flush=True)

        return np.array([loss, sloss, ulloss, uuloss, acc])

    def _eval_epoch_end(self, datasep: Separation) -> np.ndarray:
        loss = np.mean(self.losses)
        sloss = np.mean(self.sup_losses)
        ulloss = np.mean(self.unsups_lower)
        uuloss = np.mean(self.unsups_upper)

        acc = self.correct_sup / self.total_sup
        uacc = self.correct_all / self.total_all
        print(f'epoch{self.epoch} separation {datasep}, loss value {loss}, sloss {sloss}, ulloss {ulloss}, uuloss {uuloss}; acc supervised {acc}, unsupervised {uacc}', flush=True)

        return np.array([loss, sloss, ulloss, uuloss, acc])

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        best = self._criteria_vals[self._VAL_CRITERION_IDX, best_epoch, self.best_epoch_criterion_idx]
        return val > best

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['SLOSS', 'ULOSS_L', 'ULOSS_U', 'ACC']

    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')