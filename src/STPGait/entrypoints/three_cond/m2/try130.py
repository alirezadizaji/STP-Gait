from typing import Tuple

import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Label, Optim
from ....models.wifacct.gcn import GCNConv, GCNConvFC
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ..m3.try81 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# try 130 (81 ->)
## Condition concatenation on the frames, using GCN-3l network
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
            try_num=130,
            try_name="m2_gcn3l",
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
                
                self.gcn_fc = GCNConvFC(60, num_classes)
            
            def forward(self, x, edge_index):
                for m in self.gcn3l:
                    x = m(x, edge_index)
                
                x = self.gcn_fc(x, edge_index)
                return x

        gcn3l = _Module()
        return gcn3l
    
    def _get_edges(self, num_frames: int):
        return torch.from_numpy(Skeleton.get_vanilla_edges(num_frames)[0])

    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0][..., [0, 1]].to(self.conf.device) # N, M, T, V, C
        N, *_, V, C = x.size()
        x = x.reshape(N, -1, V, C)
        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1)).to(x.device)
        
        out: OUT = self.model(x, self._edge_index)
        return out