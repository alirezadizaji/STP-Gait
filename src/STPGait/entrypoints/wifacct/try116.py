from typing import Tuple

import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.ms_g3d import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .try70 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 116 (70 ->)
## Use try100 dataset
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
            try_num=116,
            try_name="wifacct_msg3d_sup_part",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=32)
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
        
        model = WiFaCCT[Model1, Model2](model1, model2, num_frames=191, num_aux_branches=3)
        return model

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        _, y, _, labeled = data
        o_main, _ = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        loss = loss_sup
        return loss