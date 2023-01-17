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
from .transformer import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

class Entrypoint(E, TrainEntrypoint[IN, OUT, float, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=GraphSkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=10, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=False,
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=2,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super().__init__(kfold, config)
    
    def get_model(self):
        model = GCNLSTMTransformer()
        return model
        
    def _model_forwarding(self, data: IN) -> OUT:
        x, y, edge_index = data
        x = self.model(x.to(self.conf.device), y.to(self.conf.device), edge_index.to(self.conf.device))
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        loss1, loss2 = x[1], x[2]
        return 0.2 * loss1 + loss2

    def _train_start(self) -> None:
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()
        self.pred: List[np.ndarray] = list()

    def _eval_iter_end(self, iter_num: int, separation: Separation, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())

    def _eval_epoch_end(self, datasep: Separation):
        print(f'epoch {self.epoch} separation {datasep} loss value {np.mean(self.losses)}', flush=True)
        return np.mean(self.losses)