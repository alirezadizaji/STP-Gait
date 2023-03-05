from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from scipy.stats import chi2_contingency
from sklearn import svm
from sklearn.metrics import multilabel_confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from ..wifacct.try74 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 115
## supervised SVM
## 5 metaclasses, condition = PS
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/cond12metaclass_PS.pkl",
                filterout_hardcases=True,
                savename="Processed_meta_PS_balanced.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)
                , num_unlabeled=500 , num_per_class=100, metaclass=True))
            )
        config = BaseConfig(
            try_num=115,
            try_name="wifacct_svm_balanced_s",
            device="cuda:0",
            eval_batch_size=12,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
    
    def get_model(self) -> nn.Module:
        model = svm.SVC(gamma='auto')
        return model
    
    def _main_phase_train(self):
        X_train = self.train_loader.dataset.X.flatten(1)
        Y_train = self.train_loader.dataset.Y
        self.model.fit(X_train, Y_train)

        X_test = self.val_loader.dataset.X.flatten(1)
        self.y_gt = self.val_loader.dataset.Y
        self.y_pred = self.model.predict(X_test)

        self.evaluation()

    def evaluation(self):
        num_classes = self.kfold._ulabels.size
        num_samples = len(self.y_pred)

        mcm = multilabel_confusion_matrix(self.y_gt, self.y_pred)
        tps, tns = mcm[:, 1, 1], mcm[:, 0, 0]
        fns, fps = mcm[:, 1, 0], mcm[:, 0, 1]

        acc = accuracy_score(self.y_gt, self.y_pred) * 100
        spec = np.mean((tns) / (tns + fps)) * 100
        sens = np.mean((tps) / (tps + fns)) * 100
        y_pred_one_hot = np.zeros((num_samples, num_classes), dtype=np.int64)
        y_pred_one_hot[np.arange(num_samples), self.y_pred] = 1
        
        f1 = f1_score(self.y_gt, self.y_pred, average='macro')
        auc = roc_auc_score(self.y_gt, y_pred_one_hot, multi_class='ovr') * 100
        rec = recall_score(self.y_gt, self.y_pred, average='macro') * 100
        pre = precision_score(self.y_gt, self.y_pred, average='macro') * 100

        observed = np.zeros((2, num_classes))
        np.add.at(observed[0], self.y_pred, 1)
        np.add.at(observed[1], self.y_gt, 1)
        _, p, *_ = chi2_contingency(observed)

        print(f'acc {acc:.2f} spec {spec:.2f} sens {sens:.2f} f1 {f1:.2f} auc {auc:.2f} p-value {p:.3f} precision {pre:.2f} recall {rec:.2f}.', flush=True)
            
        return np.array([acc, f1, sens, spec, auc, rec, pre, p])

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC', 'F1', 'Sens', 'Spec', 'AUC', 'Recall', 'Precision', 'P']

    def _calc_loss(self, x, data):
        pass
    
    def _eval_epoch_end(self, datasep):
        pass
    
    def _eval_iter_end(self, separation, iter_num, loss, x, data):
        pass

    def _eval_start(self):
        pass

    def _model_forwarding(self, data):
        pass

    def _train_epoch_end(self):
        pass

    def _train_iter_end(self, iter_num, loss, x, data):
        pass
    
    def _train_start(self):
        pass

    def best_epoch_criteria(self, best_epoch):
        pass