import os
from typing import List, Tuple
import pickle

import numpy as np
import torch

from ....config import BaseConfig, TrainingConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Separation, Label
from ....models import GCNLSTMTransformer
from ....preprocess.main import PreprocessingConfig
from .try2 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 11 (try 8 ->)
## Filter hard case samples
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ANXIOUS, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                filterout_hardcases=True,
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=11,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super(E, self).__init__(kfold, config)