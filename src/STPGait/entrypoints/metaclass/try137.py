from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Optim, Phase, Separation
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .try115 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 120
## rerun supervised SVM
## 5 metaclasses, condition = PS
class Entrypoint(E):
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
            try_num=120,
            try_name="wifacct_svm_rerun",
            device="cuda:0",
            eval_batch_size=12,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
        self.rerun: bool = False

    def set_loaders(self) -> None:
        # DO NOT SHUFFLE
        self.train_loader = DataLoader(self.kfold.train, batch_size=self.conf.training_config.batch_size, shuffle=False)
        self.val_loader = DataLoader(self.kfold.val, batch_size=self.conf.eval_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.kfold.test, batch_size=self.conf.eval_batch_size, shuffle=False)

    def _sub_run(self):  
        self.model = self.get_model()
        if isinstance(self.model, nn.Module):
            self.model.to(self.conf.device)
            self.set_optimizer(self.conf.training_config.optim_type)
        self.set_loaders()
        self._main_phase_train()
        

    def _relabel(self, loader: DataLoader):
        y_pred = []
        X = loader.dataset.X.flatten(1)
        y_pred = self.model.predict(X)

        y_pred = torch.from_numpy(y_pred)
        loader.dataset.Y[~loader.dataset.labeled] = y_pred[~loader.dataset.labeled]

    def _main_phase_train(self):
        X_train = self.train_loader.dataset.X.flatten(1)
        Y_train = self.train_loader.dataset.Y
        self.model.fit(X_train, Y_train)

        criterion = self.evaluation()
        if self.rerun:
            self.fold_test_criterion_rerun[:,self.kfold.testK] = criterion
        else:
            self.fold_test_criterion[:,self.kfold.testK] = criterion

    def run(self):
        self.fold_test_criterion: np.ndarray = np.full((len(self.criteria_names), self.kfold.K), fill_value=-np.inf)
        self.fold_test_criterion_rerun = np.full_like(self.fold_test_criterion, fill_value=-np.inf)
        print(f"@@@@@@@@@@@@ PHASE {self.conf.phase} IN PROGRESS... @@@@@@@@@@@@", flush=True)
        
        for self.current_K in range(self.kfold.K):
            with self.kfold:
                self._sub_run()
            
                # labeling
                with torch.no_grad():
                    self.model.eval()
                    for loader in [self.train_loader, self.val_loader, self.test_loader]:
                        self._relabel(loader)
                    self.rerun = True
                
                self._sub_run()
                self.rerun = False

        cv = {c: np.around(np.mean(v), decimals=4) for c, v in zip(self.criteria_names, self.fold_test_criterion)}
        print(f"@@@ Final Result on {self.kfold.K} KFold operation on test set: {cv} @@@", flush=True)

        cv = {c: np.around(np.mean(v), decimals=4) for c, v in zip(self.criteria_names, self.fold_test_criterion_rerun)}
        print(f"@@@ Final Result on rerun {self.kfold.K} KFold operation on test set: {cv} @@@", flush=True)
        