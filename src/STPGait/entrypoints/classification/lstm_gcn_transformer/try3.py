from typing import Tuple

import torch
from torch.utils.data import DataLoader

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim
from ....models import GCNLSTMTransformerV2
from ....preprocess.main import PreprocessingConfig
from .try2 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 3 (try 2 ->)
## K=5, Test set have been removed indeed (Val and Test are same)
## Temporal edges removed
## LSTM state initialized randomly only at the first iteration of first epoch
## LSTM state updated after each forwarding
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0),
                load_dir="../../Data/output_1.pkl",
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
        TrainEntrypoint.__init__(self, kfold, config)

    
    def get_model(self):
        model = GCNLSTMTransformerV2(get_gcn_edges= lambda T: torch.from_numpy(Skeleton.get_vanilla_edges(T)[0]))
        return model

    def set_loaders(self) -> None:
        self.train_loader = DataLoader(self.kfold.train, batch_size=self.conf.training_config.batch_size, shuffle=self.conf.training_config.shuffle_training, drop_last=True)
        self.val_loader = DataLoader(self.kfold.val, batch_size=self.conf.eval_batch_size, drop_last=True)
        self.test_loader = DataLoader(self.kfold.test, batch_size=self.conf.eval_batch_size, drop_last=True)