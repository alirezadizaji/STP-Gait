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
from ...enums import Separation, Optim, Label
from ...models.wifacct import WiFaCCT
from ...models.wifacct.ms_g3d import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from ..wifacct.try116 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 121 (116 -> )
## Condition editable
## 3 classes = healthy, hypokinetic, ataxic
## MSG3D
class Entrypoint(E):
    def __init__(self) -> None:
        self.cond = "DTM"
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
            try_num=121,
            try_name="cond3class_MSG3D" + self.cond,
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None
    
    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size

        num_point = 25
        num_person=1
        num_gcn_scales=4
        num_g3d_scales=2

        model1 = Model1(
            num_point=num_point,
            num_person=num_person,
            num_gcn_scales=num_gcn_scales,
            num_g3d_scales=num_g3d_scales)
        model2 = Model2(
            num_class=num_classes,
            num_gcn_scales=num_gcn_scales,
            num_g3d_scales=num_g3d_scales)
        
        if self.cond == 'DTM':
            num_frames = 159
        elif self.cond == 'EC':
            num_frames = 232
        elif self.cond == 'PS':
            num_frames = 183
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=num_frames, num_aux_branches=3)
        return model

