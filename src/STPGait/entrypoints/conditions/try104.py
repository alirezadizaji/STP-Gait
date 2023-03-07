from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim, Label
from ...preprocess.main import PreprocessingConfig
from ...models.wifacct import WiFaCCT
from ...models.wifacct.mst_gcn import Model1, Model2
from ..train import TrainEntrypoint
from ..wifacct.try78 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 104 (78 -> )
## MST-GCN 3 block
## Condition editable
## 3 classes = healthy, hypokinetic, ataxic
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
            try_num=104,
            try_name="cond3class_MSTGCN_sup_" + self.cond,
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        
        in_channels = 2
        num_point = 25
        num_person=1
        num_classes = self.kfold._ulabels.size
        basic_channels = 60

        cfgs1 = {
        'num_class': num_classes,
        'num_point': 25,
        'num_person': 1,
        'block_args': [[2, basic_channels, False, 1],
                       [basic_channels, basic_channels, True, 1]],
        # 'graph': 'graph.kinetics.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'kernel_size': 9,
        'block_type': 'ms',
        'reduct_ratio': 2,
        'expand_ratio': 0,
        't_scale': 4,
        'layer_type': 'sep',
        'act': 'relu',
        's_scale': 4,
        'atten': 'stcja',
        'bias': True,
        # 'parts': parts
        }
        model1 = Model1(**cfgs1)

        cfgs2 = {
        'A': model1.A,
        'num_class': num_classes,
        'num_point': 25,
        'num_person': 1,
        'block_args': [[basic_channels, basic_channels, True, 1]],
        # 'graph': 'graph.kinetics.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'kernel_size': 9,
        'block_type': 'ms',
        'reduct_ratio': 2,
        'expand_ratio': 0,
        't_scale': 4,
        'layer_type': 'sep',
        'act': 'relu',
        's_scale': 4,
        'atten': 'stcja',
        'bias': True,
        # 'parts': parts
        }
        model2 = Model2(**cfgs2)
        if self.cond == 'DTM':
            num_frames = 317
        elif self.cond == 'EC':
            num_frames = 464
        elif self.cond == 'PS':
            num_frames = 366
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=num_frames, num_aux_branches=3)
        return model