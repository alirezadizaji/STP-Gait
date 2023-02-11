from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Label
from ....models import GCNLSTMTransformerV2
from ....preprocess.main import PreprocessingConfig
from .try7 import Entrypoint as E
from ...train import TrainEntrypoint

# try 37 (try 7 ->)
## Remove cnn part
## Filterout hard cases
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ATAXIC, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=37,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    def get_model(self):
        model = GCNLSTMTransformerV2(num_classes=3, cnn_conf=None, get_gcn_edges= lambda T: torch.from_numpy(Skeleton.get_vanilla_edges(T)[0]))
        return model