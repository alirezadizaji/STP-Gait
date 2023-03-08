import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Label, Optim
from ....models.multicond import AggMode, MultiCond
from ....models.wifacct.vivit import Model1
from ....preprocess.main import PreprocessingConfig
from .try127 import Entrypoint as E
from ...train import TrainEntrypoint

# try 129 (127 ->)
## Use Attention mode
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
            try_num=129,
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
        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.model1 = Model1(d_model=d, nhead=8, n_enc_layers=2)

            def forward(self, x):
                x = self.model1(x)
                x = x.mean((1, 2))
                return x

        m = _Module()
        model = MultiCond[_Module](m, fc_hidden_num=[72, 60], agg_mode=AggMode.ATT, num_classes=num_classes, z_dim=72)
        return model