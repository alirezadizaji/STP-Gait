from typing import List, Tuple

import numpy as np
import torch

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim, Label
from ...models.wifacct import WiFaCCT
from ...models.wifacct.vivit import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from ..wifacct.try76 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 105 (76 -> )
## Vivit block
## Condition editable
## 3 classes = healthy, hypokinetic, ataxic
class Entrypoint(E):
    def __init__(self) -> None:
        self.cond = "PS"
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True,
                remove_labels= [Label.ANXIOUS, Label.SENSORY_ATAXIC, Label.PARETIC,
                Label.HYPOKENITC_FRONTAL, Label.ANTALGIC, Label.DYSKINETIC, 
                Label.FUNCTIONAL, Label.MOTOR_COGNITIVE, Label.SPASTIC]),
                load_dir="../../Data/sepCondAllClass/cond12class_"+ self.cond +".pkl",
                filterout_hardcases=True,
                savename="processed_12c_"+ self.cond +".pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=105,
            try_name="cond3class_Vivit_sup_" + self.cond,
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    @property
    def frame_size(self):
        return 300

    def get_model(self):
        num_classes = self.kfold._ulabels.size

        d = 60
        model1 = Model1(d_model=d, nhead=6, n_enc_layers=2)
        model2 = Model2(num_classes, d_model=d, nhead=6, n_enc_layers=1)
        
        model = WiFaCCT[Model1, Model2](model1, model2, num_aux_branches=3)
        return model