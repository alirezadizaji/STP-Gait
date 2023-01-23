from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Label
from ....models import GCNLSTMTransformerV2
from ....preprocess.main import PreprocessingConfig
from .try3 import Entrypoint as E
from .try2 import Entrypoint as EE

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 9 (try 3 ->)
## Use only ataxic-hypokinetic_frontal-healthly
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ANXIOUS, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=9,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super(E, self).__init__(kfold, config)