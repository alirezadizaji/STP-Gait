from typing import List, Tuple

import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.multicond import MultiCond
from ....models.wifacct.ms_g3d import Model1
from ....preprocess.main import PreprocessingConfig
from .try81 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 124 (81 ->)
## Three-branch condition based training, using three MS-G3D sub-block
## Mode: Take average of latent space
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
            try_num=124,
            try_name="threecond_msg3d",
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

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Model1(
                    num_point=num_point,
                    num_person=num_person,
                    num_gcn_scales=num_gcn_scales,
                    num_g3d_scales=num_g3d_scales)
            
            def forward(self, x):
                x = self.model(x)
                x = x.mean((1, 2))
                return x

        model = _Module()
        model = MultiCond[_Module](model, fc_hidden_num=[60, 60], num_classes=num_classes)
        return model

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C        
        cond_mask = data[4].permute(1, 0).to(self.conf.device)
        x = x.permute(1, 0, 2, 3, 4)
        inps = list()
        for x_ in x:
            inps.append((x_,))

        out: OUT = self.model(cond_mask, inps)
        return out