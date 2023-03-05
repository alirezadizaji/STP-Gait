from typing import List, Tuple

import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.lstm_gcn import Model
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .dynamic_coef.try113 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 114 (113 ->)
## Supervised training
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
            try_num=114,
            try_name="wifacct_lstm_gcn_sup_part",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=32)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        _, y, _, labeled = data
        o_main, _ = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        loss = loss_sup
        return loss