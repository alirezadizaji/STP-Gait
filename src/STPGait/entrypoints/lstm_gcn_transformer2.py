from typing import Tuple

import torch

from ..config import BaseConfig, TrainingConfig
from ..context import Skeleton
from ..data.read_gait_data import ProcessingGaitConfig
from ..dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ..enums import Optim
from ..models import GCNLSTMTransformerV2
from ..preprocess.main import PreprocessingConfig
from .train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# I2 (I1 ->)
## K=5, Test set have been removed indeed (Val and Test are same)
## Temporal edges removed
## LSTM state initialized randomly only at the first iteration of first epoch
## LSTM state updated after each forwarding
class Entrypoint(TrainEntrypoint[IN, OUT, float, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=3,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super().__init__(kfold, config)

    
    def get_model(self):
        model = GCNLSTMTransformerV2(get_gcn_edges= lambda T: Skeleton.get_vanilla_edges(T))
        return model