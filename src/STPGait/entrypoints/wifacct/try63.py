import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
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

# try 63
## Semi-supervised training using GCN-3l
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
            try_num=63,
            try_name="wifacct_gcn",
            device="cuda:0",
            phase="EVAL",
            eval_batch_size=32,
            save_log_in_file=True,
            model_runtime_recording=True,
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
        o_main, o_aux = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        y1d = o_main.argmax(1).detach().unsqueeze(1).repeat(1, o_aux.size(1)).flatten()
        o_aux = o_aux.flatten(0, 1)
        loss_unsup = -torch.mean(o_aux[torch.arange(y1d.size(0)), y1d])

        loss = 0.2 * loss_unsup
        if not torch.isnan(loss_sup):
            loss = loss + loss_sup
        return loss

    def _train_start(self) -> None:
        self.y_pred = list()
        self.y_gt = list()
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        _, y, _, labeled = data
        x_probs = x[0][labeled]
        y_pred = x_probs.argmax(-1)

        self.y_pred += y_pred.tolist()
        self.y_gt += y[labeled].tolist()

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, separation: Separation, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())
        
        _, y, _, labeled = data
        x_probs = x[0][labeled]
        y_pred = x_probs.argmax(-1)
        
        self.y_pred += y_pred.tolist()
        self.y_gt += y[labeled].tolist()

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
        rec = recall_score(self.y_gt, self.y_pred, average='macro') * 100
        pre = precision_score(self.y_gt, self.y_pred, average='macro') * 100

        observed = np.zeros((2, num_classes))
        np.add.at(observed[0], self.y_pred, 1)
        np.add.at(observed[1], self.y_gt, 1)
        _, p, *_ = chi2_contingency(observed)

        print(f'epoch{self.epoch} loss value {loss:.2f} acc {acc:.2f} spec {spec:.2f} sens {sens:.2f} f1 {f1:.2f} auc {auc:.2f} p-value {p:.3f} recall {rec:.2f} precision {pre:.2f}', flush=True)
        return np.array([loss, acc, f1, sens, spec, auc, rec, pre, p])

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
        rec = recall_score(self.y_gt, self.y_pred, average='macro') * 100
        pre = precision_score(self.y_gt, self.y_pred, average='macro') * 100

        observed = np.zeros((2, num_classes))
        np.add.at(observed[0], self.y_pred, 1)
        np.add.at(observed[1], self.y_gt, 1)
        _, p, *_ = chi2_contingency(observed)

        if datasep == Separation.TEST:
            res = np.zeros((num_classes, num_classes))
            np.add.at(res, (self.y_gt, self.y_pred), 1)
            res = res * 100 / res.sum(1)[:, np.newaxis]

            df = pd.DataFrame(data=res, index=self.kfold._ulabels, columns=self.kfold._ulabels)
            ax = sns.heatmap(df, annot=True, vmin=0, vmax=100, fmt=".1f", annot_kws={"fontsize":12})
            for t in ax.texts: t.set_text(t.get_text() + "%")
            
            plt.xticks(rotation = 20)
            plt.yticks(rotation = 45)
            plt.xlabel('GT')
            plt.ylabel('Pred')
            
            save_dir = os.path.join(self.conf.save_dir, "cm", f"test{self.kfold.testK}", "img.png")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            plt.savefig(save_dir, dpi=600, bbox_inches='tight')

        print(f'epoch{self.epoch} separation {datasep} loss value {loss:.2f} acc {acc:.2f} spec {spec:.2f} sens {sens:.2f} f1 {f1:.2f} auc {auc:.2f} p-value {p:.3f} precision {pre:.2f} recall {rec:.2f}.', flush=True)
            
        return np.array([loss, acc, f1, sens, spec, auc, rec, pre, p])

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        best = self._criteria_vals[self._VAL_CRITERION_IDX, best_epoch, self.best_epoch_criterion_idx]
        return val > best

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC', 'F1', 'Sens', 'Spec', 'AUC', 'Recall', 'Precision', 'P']

    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')