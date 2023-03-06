from typing import List, Tuple

import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.multicond import MultiCond
from ....models.wifacct.vivit import Model1
from ....preprocess.main import PreprocessingConfig
from .try81 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 127 (81 ->)
## Three-branch condition based training, using three ViViT
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
            try_num=127,
            try_name="threecond_vivit",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size

        d = 72
        model1 = Model1(d_model=d, nhead=8, n_enc_layers=2)
        model = MultiCond[Model1](model1, fc_hidden_num=[60, 60], num_classes=num_classes)
        return model

    @property
    def frame_size(self):
        return 360

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C        
        cond_mask = data[4].permute(1, 0).to(self.conf.device)
        x = x.permute(1, 0, 2, 3, 4)
        inps = list()
        for x_ in x:
            inps.append((x_[:, :self.frame_size]))

        out: OUT = self.model(cond_mask, inps)
        return out