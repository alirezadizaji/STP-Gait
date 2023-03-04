from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from ...config import BaseConfig, TrainingConfig
from ...context import Skeleton
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.st_gcn import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from ..wifacct.try74 import Entrypoint as E

# try 100
## supervised ST-GCN
## 5 metaclasses, condition = PS
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/cond12metaclass_PS.pkl",
                filterout_hardcases=True,
                savename="Processed_meta_PS_balanced.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)
                , num_unlabeled=500 , num_per_class=100, metaclass=True))
            )
        config = BaseConfig(
            try_num=100,
            try_name="wifacct_stgcn_balanced_s",
            device="cuda:0",
            eval_batch_size=12,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
    
    def get_model(self) -> nn.Module:
        
        in_channels = 2
        num_point = 25
        num_person=1
        num_classes = self.kfold._ulabels.size

        model1 = Model1(in_channels, num_classes, True, None)
        model2 = Model2(model1.A, num_classes, True, None)
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=188, num_aux_branches=5)
        return model