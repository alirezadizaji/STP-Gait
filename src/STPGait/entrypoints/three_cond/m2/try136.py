from typing import Tuple

import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.others.mst_gcn.model.AEMST_GCN import Model
from ....models.wifacct import WiFaCCT
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ..m3.try81 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 136 (81 ->)
## Condition concatenation on the frames, using MST-GCN network
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
            try_num=136,
            try_name="m2_mstgcn",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=False,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self):
        num_classes = self.kfold._ulabels.size
        basic_channels = 32
        cfgs = {
        'num_class': num_classes,
        'num_point': 25,
        'num_person': 1,
        'block_args': [[2, basic_channels, False, 1],
                       [basic_channels, basic_channels, True, 1], [basic_channels, basic_channels, True, 1]],
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
        }
        model = Model(**cfgs)
        return model
    
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C
        N, *_, V, C = x.size()
        x = x.reshape(N, -1, V, C)
        x = torch.permute(x, (0, 3, 1, 2)) # B=batch size, C=2, T, V
        x = torch.unsqueeze(x, dim=-1)
        
        out: OUT = self.model(x)
        return out
    
    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        y = data[1]
        idx = torch.arange(y.numel())
        m = torch.nn.LogSoftmax(dim=1)
        output = m(x[1])
        loss = -torch.mean(output[idx, y]) #CE
        return loss