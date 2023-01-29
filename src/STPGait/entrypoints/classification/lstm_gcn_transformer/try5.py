from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Label
from ....preprocess.main import PreprocessingConfig
from .try2 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 5 (try 2 ->)
## Use only anxious-hypokinetic_frontal-healthly
## Use 5Fold operation
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ATAXIC, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=5,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(kfold, config)