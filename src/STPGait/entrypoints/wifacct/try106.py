from typing import List, Tuple

import torch

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonMultiPklKFoldOperator, SkeletonKFoldMultiPklConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Optim
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .try63 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 106 (68 ->)
## Use multi pickle skeleton kfold operator
class Entrypoint(E):
    def __init__(self) -> None:
        base = "../../Data/5foldFinal"
        kfold = GraphSkeletonMultiPklKFoldOperator(
            config=SkeletonKFoldMultiPklConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir=[f"{base}/test_{i}.pkl" for i in range(5)],
                unite_save_dir="../../Data/5foldFinal.pkl",
                filterout_hardcases=True,
                savename="processed_5foldFinal.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120),metaclass=True)
                )
            )
        config = BaseConfig(
            try_num=106,
            try_name="wifacct_gcn_sup_part",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        _, y, _, labeled = data
        o_main, _ = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        loss = loss_sup
        return loss