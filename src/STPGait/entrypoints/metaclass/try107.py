from typing import List, Tuple

import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Separation, Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.mst_gcn import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from ..wifacct.try78 import Entrypoint as E

# try 107
## supervised MST-GCN
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
            try_num=107,
            try_name="wifacct_mstgcn_balanced_s",
            device="cuda:0",
            eval_batch_size=12,
            save_log_in_file=False,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=12)
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
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=385, num_aux_branches=3)
        return model

