from typing import List, Tuple

import numpy as np
import torch

from .....config import BaseConfig, TrainingConfig
from .....data.read_gait_data import ProcessingGaitConfig
from .....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from .....enums import Label, Optim, Separation
from .....models.others.msg3d.model import Model
from .....preprocess.main import PreprocessingConfig
from .try57 import Entrypoint as E
from ....train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# Try 59 (57 ->)
# Use ataxic-hypokinetic_frontal-healthy labels
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True, remove_labels=[Label.ANXIOUS, Label.SENSORY_ATAXIC, Label.PARETIC]),
                filterout_hardcases=True,
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=59,
            try_name="ms_g3d",
            device="cuda:0",
            eval_batch_size=12,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
        )
        TrainEntrypoint.__init__(self, kfold, config)
