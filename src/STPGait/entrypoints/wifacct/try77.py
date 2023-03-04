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
from ...models.wifacct.mst_gcn import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .try63 import Entrypoint as E


# try 77
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=False),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=77,
            try_name="wifacct_mstgcn_b3",
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
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=417, num_aux_branches=3)
        return model

    def _model_forwarding(self, data):
        x = data[0][..., [0, 1]].to(self.conf.device) # Use X-Y features

        out = self.model(x, m1_args=dict(), m2_args=dict())
        
        return out

    def _calc_loss(self, x, data):
        _, y, _, labeled = data
        o_main, o_aux = x

        m = torch.nn.LogSoftmax(dim=1)
        o_main = m(o_main)
        o_aux = m(o_aux)
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        y1d = o_main.argmax(1).detach().unsqueeze(1).repeat(1, o_aux.size(1)).flatten()
        o_aux = o_aux.flatten(0, 1)
        loss_unsup = -torch.mean(o_aux[torch.arange(y1d.size(0)), y1d])

        loss = 0.2 * loss_unsup
        if not torch.isnan(loss_sup):
            loss = loss + loss_sup

        return loss