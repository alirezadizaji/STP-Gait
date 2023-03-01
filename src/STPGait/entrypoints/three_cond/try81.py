from typing import List, Tuple

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, f1_score, roc_auc_score, accuracy_score
import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim
from ...models.multicond import MultiCond
from ...models.wifacct.gcn import GCNConv
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 81
## Three-branch condition based training, using three GCN-3l network
## Mode: Take average of latent space
## Semi-supervised training using GCN-3l
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonCondKFoldOperator(
            config=SkeletonCondKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/cond12class.pkl",
                filterout_hardcases=True,
                savename="processed12cls_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)),
                min_num_valid_cond=3)
            )
        config = BaseConfig(
            try_num=81,
            try_name="threecond_gcn",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size
        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
            
                self.gcn3l = nn.ModuleList([
                    GCNConv(2, 60),
                    GCNConv(60, 60),
                    GCNConv(60, 60)])
            
            def forward(self, x, edge_index):
                for m in self.gcn3l:
                    x = m(x, edge_index)
                
                return x
        
        gcn3l = _Module()
        model = MultiCond[_Module](gcn3l, fc_hidden_num=[60, 60], num_classes=num_classes)
        return model
    
    def _get_edges(self, num_frames: int):
        return torch.from_numpy(Skeleton.get_vanilla_edges(num_frames)[0])

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C
        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1)).to(x.device)
        
        cond_mask = data[4].permute(1, 0).to(self.conf.device)
        x = x.permute(1, 0, 2, 3, 4)
        inps = list()
        for x_ in x:
            inps.append((x_, self._edge_index))


        out: OUT = self.model(cond_mask, inps)
        return out

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        y = data[1]
        loss = -torch.mean(x[torch.arange(y.numel()), y])

        return loss

    def _train_start(self) -> None:
        self.y_pred = list()
        self.y_gt = list()
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        y = data[1]
        x_probs = x
        y_pred = x_probs.argmax(-1)

        self.y_pred += y_pred.tolist()
        self.y_gt += y.tolist()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())
        
        y = data[1]
        x_probs = x
        y_pred = x_probs.argmax(-1)
        
        self.y_pred += y_pred.tolist()
        self.y_gt += y.tolist()

    def _train_epoch_end(self) -> np.ndarray:
        num_classes = self.kfold._ulabels.size
        num_samples = len(self.y_pred)
        loss = np.mean(self.losses)
        
        mcm = multilabel_confusion_matrix(self.y_gt, self.y_pred)
        tps, tns = mcm[:, 1, 1], mcm[:, 0, 0]
        fns, fps = mcm[:, 1, 0], mcm[:, 0, 1]

        acc = accuracy_score(self.y_gt, self.y_pred) * 100
        spec = np.mean((tns) / (tns + fps)) * 100
        sens = np.mean((tps) / (tps + fns)) * 100
        
        y_pred_one_hot = np.zeros((num_samples, num_classes), dtype=np.int64)
        y_pred_one_hot[np.arange(num_samples), self.y_pred] = 1

        f1 = f1_score(self.y_gt, self.y_pred, average='macro') * 100
        auc = roc_auc_score(self.y_gt, y_pred_one_hot, multi_class='ovr') * 100
        print(f'epoch{self.epoch} loss value {loss:.2f} acc {acc:.2f} spec {spec:.2f} sens {sens:.2f} f1 {f1:.2f} auc {auc:.2f}', flush=True)

        return np.array([loss, acc, f1, sens, spec, auc])

    def _eval_epoch_end(self, datasep: Separation) -> np.ndarray:
        num_classes = self.kfold._ulabels.size
        num_samples = len(self.y_pred)
        loss = np.mean(self.losses)
        
        mcm = multilabel_confusion_matrix(self.y_gt, self.y_pred)
        tps, tns = mcm[:, 1, 1], mcm[:, 0, 0]
        fns, fps = mcm[:, 1, 0], mcm[:, 0, 1]

        acc = accuracy_score(self.y_gt, self.y_pred) * 100
        spec = np.mean((tns) / (tns + fps)) * 100
        sens = np.mean((tps) / (tps + fns)) * 100
        y_pred_one_hot = np.zeros((num_samples, num_classes), dtype=np.int64)
        y_pred_one_hot[np.arange(num_samples), self.y_pred] = 1

        f1 = f1_score(self.y_gt, self.y_pred, average='macro') * 100
        auc = roc_auc_score(self.y_gt, y_pred_one_hot, multi_class='ovr') * 100
        print(f'epoch{self.epoch} separation {datasep} loss value {loss:.2f} acc {acc:.2f} spec {spec:.2f} sens {sens:.2f} f1 {f1:.2f} auc {auc:.2f}', flush=True)

        return np.array([loss, acc, f1, sens, spec, auc])

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        best = self._criteria_vals[self._VAL_CRITERION_IDX, best_epoch, self.best_epoch_criterion_idx]
        return val > best

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC', 'F1', 'Sens', 'Spec', 'AUC']

    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')