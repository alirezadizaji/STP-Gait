from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

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
            save_log_in_file=False,
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
        Y_test = self.val_loader.dataset.Y
        Y_pred = self.model.predict(X_test)

        self.evaluation(Y_test, Y_pred)

    def evaluation(self, y_true, y_pred):
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        # F1 score
        f1score = f1_score(y_true, y_pred, average='macro') # macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

        print(f'Mean accuracy score: {accuracy:.3}')
        print(f'Mean F1 score (macro): {f1score:.3}')

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