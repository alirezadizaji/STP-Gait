from typing import List, Tuple

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, roc_auc_score, accuracy_score
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.gcn import GCNConv, GCNConvFC
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 90
## supervised training using GCN-3l
## Seperating all conditions data
## Condition editable
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        self.cond = "PS"
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/sepCondAllClass/cond12class_"+ self.cond +".pkl",
                filterout_hardcases=True,
                savename="processed_12c_"+ self.cond +".pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=90,
            try_name="cond12class_" + self.cond,
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=False,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size
        num_shared_gcn=2
        dim_node=2
        dim_hidden=60

        class M1(nn.Module):
            def __init__(self):
                super().__init__()
                self.ml = nn.ModuleList(
                    [GCNConv(dim_node, dim_hidden)] +
                    [GCNConv(dim_hidden, dim_hidden) for _ in range(num_shared_gcn - 1)])
            
            def forward(self, x, edge_index):
                for l in self.ml:
                    x = l(x, edge_index)
                
                return x

        model1 = M1()
        model2 = GCNConvFC(dim_hidden, num_classes)
        
        model = WiFaCCT[nn.ModuleList, GCNConvFC](model1, model2, num_aux_branches=5)
        return model
    
    def _get_edges(self, num_frames: int):
        return torch.from_numpy(Skeleton.get_vanilla_edges(num_frames)[0])

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # Use X-Y features

        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1)).to(x.device)

        out: OUT = self.model(x, m1_args=dict(edge_index=self._edge_index), m2_args=dict(edge_index=self._edge_index))
        return out

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        _, y, _, labeled = data
        o_main, _ = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        loss = loss_sup
        return loss

    def _train_start(self) -> None:
        self.correct = self.total = 0
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        _, y, _, labeled = data
        x_probs = x[0][labeled]
        y_pred = x_probs.argmax(-1)

        self.correct += torch.sum(y_pred == y[labeled]).item()
        self.total += y[labeled].numel()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())
        
        _, y, _, labeled = data
        x_probs = x[0][labeled]
        y_pred = x_probs.argmax(-1)
        
        self.correct += torch.sum(y_pred == y[labeled]).item()
        self.total += y[labeled].numel()

    def _train_epoch_end(self) -> np.ndarray:
        loss = np.mean(self.losses)
        acc = self.correct / (self.total + 1e-4)
        print(f'epoch{self.epoch} loss value {loss} acc {acc}', flush=True)
        return np.array([loss, acc])

    def _eval_epoch_end(self, datasep: Separation) -> np.ndarray:
        acc = self.correct / (self.total + 1e-4)
        loss = np.mean(self.losses)
        print(f'epoch {self.epoch} separation {datasep} loss value {loss} acc {acc}', flush=True)
        return np.array([loss, acc])

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        best = self._criteria_vals[self._VAL_CRITERION_IDX, best_epoch, self.best_epoch_criterion_idx]
        return val > best

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC']

    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')