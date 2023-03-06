import os
from tqdm import tqdm
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim, Phase, Separation
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ...metaclass.try100 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 119
## supervised STGCN
## 5 metaclasses, condition = PS
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=False),
                load_dir="../../Data/cond12metaclass_PS.pkl",
                filterout_hardcases=True,
                savename="Processed_meta_PS_balanced.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)
                , num_unlabeled=500 , num_per_class=100, metaclass=True))
            )
        config = BaseConfig(
            try_num=119,
            try_name="wifacct_stgcn_rerun",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
        self._start_ul_epoch: int = 10
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

        if self.conf.phase == Phase.TRAIN:
            self._main_phase_train()
        elif self.conf.phase == Phase.EVAL:
            self._main_phase_eval()
        

    def _relabel(self, loader: DataLoader):
        y_pred = []
        for data in loader:
            for i in range(len(data)):
                if isinstance(data[i], torch.Tensor):
                    data[i] = data[i].to(self.conf.device)
            y_pred += self._model_forwarding(data)[0].argmax(1).cpu().numpy().tolist()

        y_pred = torch.Tensor(y_pred).long()
        loader.dataset.Y[~loader.dataset.labeled] = y_pred[~loader.dataset.labeled]

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        _, y, _, labeled = data
        o_main, o_aux = x
        if self.rerun:
            labeled = torch.ones_like(y, dtype=np.bool)

        m = torch.nn.LogSoftmax(dim=1)
        o_main = m(o_main)
        o_aux = m(o_aux)

        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        return loss_sup

    def run(self):
        self.fold_test_criterion: np.ndarray = np.full((len(self.criteria_names), self.kfold.K), fill_value=-np.inf)
        self.fold_test_criterion_rerun = np.full_like(self.fold_test_criterion, fill_value=-np.inf)
        print(f"@@@@@@@@@@@@ PHASE {self.conf.phase} IN PROGRESS... @@@@@@@@@@@@", flush=True)
        
        for self.current_K in range(self.kfold.K):
            with self.kfold:
                self._sub_run()
            
                files = os.listdir(self.weight_save_dir)
                if len(files) != 1:
                    print(f"@@@ WARNING: {self.weight_save_dir} directory should have exactly one saved file; got {len(files)} instead.", flush=True)
                
                self.epoch = int(files[0])
                self.model.load_state_dict(torch.load(self._get_weight_save_path(self.epoch), map_location=self.conf.device))

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
        
    def _main_phase_train(self):
        num_epochs = self.conf.training_config.num_epochs
        best_epoch = None
        for self.epoch in tqdm(range(num_epochs)):
            
            self._train_for_one_epoch()
            
            criteria = self._validate_for_one_epoch(Separation.VAL, self.val_loader)
            self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch] = criteria
            val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
            
            # save only best epoch in terms of validation accuracy             
            if best_epoch is None or self.best_epoch_criteria(best_epoch):
                if best_epoch is not None:
                    self._remove_model_weight(best_epoch)
                self._save_model_weight()
                best_epoch = self.epoch
                print(f"### Best epoch changed to {best_epoch} criteria {val} ###", flush=True)

            # Visualize metrics
            if self.epoch % 10 == 0:
                self._visualize()

            # check early stopping if necessary
            if self.conf.training_config.early_stop is not None \
                    and self._early_stopping(best_epoch, self.conf.training_config.early_stop):
                print(f"*** EARLY STOPPING: epoch No. {self.epoch}, best epoch No. {best_epoch} ***", flush=True)
                break

        # evaluate best epoch on test set
        self.epoch = best_epoch
        self.model.load_state_dict(torch.load(self._get_weight_save_path(self.epoch), map_location=self.conf.device))
        
        criteria = self._validate_for_one_epoch(Separation.TEST, self.test_loader)
        test = criteria[self.best_epoch_criterion_idx]
        if self.rerun:
            self.fold_test_criterion_rerun[:, self.kfold.testK] = criteria
        else:
            self.fold_test_criterion[:, self.kfold.testK] = criteria
        
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        
        print(f"@@@@@ Best criteria ValK {self.kfold.valK} epoch {self.epoch} @@@@@\n---> criterion: {self.criteria_names[self.best_epoch_criterion_idx]}, val: {val:.2f}, test: {test:.2f} <----", flush=True)

    def _main_phase_eval(self):
        files = os.listdir(self.weight_save_dir)
        if len(files) != 1:
            print(f"@@@ WARNING: {self.weight_save_dir} directory should have exactly one saved file; got {len(files)} instead.", flush=True)
        
        self.epoch = int(files[0])
        self.model.load_state_dict(torch.load(self._get_weight_save_path(self.epoch), map_location=self.conf.device))
        
        criteria = self._validate_for_one_epoch(Separation.VAL, self.val_loader)
        self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch] = criteria
        val = self._criteria_vals[self._VAL_CRITERION_IDX, self.epoch, self.best_epoch_criterion_idx]
        
        criteria = self._validate_for_one_epoch(Separation.TEST, self.test_loader)
        if self.rerun:
            self.fold_test_criterion_rerun[:, self.kfold.testK] = criteria
        else:
            self.fold_test_criterion[:, self.kfold.testK] = criteria
        test = criteria[self.best_epoch_criterion_idx]
        
        print(f"@@@@@ Best criteria ValK {self.kfold.valK} epoch {self.epoch} @@@@@\n---> criterion: {self.criteria_names[self.best_epoch_criterion_idx]}, val: {val:.2f}, test: {test:.2f} <----", flush=True)