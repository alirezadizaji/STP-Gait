from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Label, Optim
from ....models.others.msg3d.model import Model
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ..m3.try81 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 131 (81 ->)
## Condition concatenation on the frames, using MS-G3D network
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonCondKFoldOperator(
            config=SkeletonCondKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True,
                remove_labels= [Label.ANXIOUS, Label.SENSORY_ATAXIC, Label.PARETIC,
                Label.HYPOKENITC_FRONTAL, Label.ANTALGIC, Label.DYSKINETIC, 
                Label.FUNCTIONAL, Label.MOTOR_COGNITIVE, Label.SPASTIC]),
                load_dir="../../Data/cond12class.pkl",
                filterout_hardcases=True,
                savename="processed12cls_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)),
                min_num_valid_cond=3)
            )
        config = BaseConfig(
            try_num=131,
            try_name="m2_ms_g3d",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self):
        num_classes = self.kfold._ulabels.size
        model = Model(
            num_class=num_classes,
            num_point=25,
            num_person=1,
            num_gcn_scales=4,
            num_g3d_scales=2)
        return model
    
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C
        N, *_, V, C = x.size()
        x = x.reshape(N, -1, V, C)
        x = x.permute(0, 3, 1, 2)[..., None] # B, C, T, V, M
        x = self.model(x)
        return x