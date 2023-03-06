import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.multicond import AggMode, MultiCond
from ....models.wifacct.gcn import GCNConv
from ....preprocess.main import PreprocessingConfig
from .try81 import Entrypoint as E
from ...train import TrainEntrypoint

# try 86 (81 ->)
## Use concatentation mode
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
            try_num=86,
            try_name="threecond_gcn",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size
        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
            
                self.gcn3l = nn.ModuleList([
                    GCNConv(2, 60),
                    GCNConv(60, 60),
                    GCNConv(60, 60)])
            
            def forward(self, x, edge_index):
                for m in self.gcn3l:
                    x = m(x, edge_index)
                
                return x
        
        gcn3l = _Module()
        model = MultiCond[_Module](gcn3l, fc_hidden_num=[180, 60], agg_mode=AggMode.CAT, num_classes=num_classes)
        return model