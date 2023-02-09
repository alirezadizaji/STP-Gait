from typing import List, Tuple

import numpy as np
import torch

from .....config import BaseConfig, TrainingConfig
from .....data.read_gait_data import ProcessingGaitConfig
from .....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from .....enums import Label, Optim, Separation
from .....models.others.vivit import ViViT, Encoder1
from .....preprocess.main import PreprocessingConfig
from ....train import TrainEntrypoint
from .try48 import Entrypoint as E


# Try 55 (48 ->)
# Use anxious-hypokinetic_frontal-healthy labels
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True, remove_labels=[Label.ATAXIC, Label.SENSORY_ATAXIC, Label.PARETIC]),
                filterout_hardcases=True,
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=55,
            try_name="vivit_m4_t40_f400",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)
