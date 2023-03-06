from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.others.vivit import ViViT, Encoder1
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ..m3.try81 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 132 (81 ->)
## Condition concatenation on the frames, using ViViT network
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonCondKFoldOperator(
            config=SkeletonCondKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/cond12class.pkl",
                filterout_hardcases=True,
                savename="processed12cls_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)),
                min_num_valid_cond=3)
            )
        config = BaseConfig(
            try_num=132,
            try_name="m2_vivit",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self):
        encoder = Encoder1(72, 6, 3)
        num_classes = self.kfold._ulabels.size
        model = ViViT(num_classes, encoder)
        return model
    
    @property
    def frame_size(self):
        return 360
    
    @property
    def chunk_size(self):
        return 10

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., :self.frame_size, :, [0, 1]].to(self.conf.device) # N, M, T, V, C
        N, *_, V, C = x.size()
        x = x.reshape(N, -1, V, C)
        x = torch.stack(x.split(self.chunk_size, 1), dim=3) # B, T1, V, T2, C
        x = x.flatten(3).to(self.conf.device)  # B, T1, V, D
        x = self.model(x)
        return x