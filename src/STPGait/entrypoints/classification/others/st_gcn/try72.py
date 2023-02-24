from typing import List, Tuple

import numpy as np
import torch

from .....config import BaseConfig, TrainingConfig
from .....data.read_gait_data import ProcessingGaitConfig
from .....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from .....enums import Optim, Separation
from .....models.others.st_gcn.st_gcn import Model
from .....preprocess.main import PreprocessingConfig
from ....train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = torch.Tensor

# Try72 ST-GCN - 12 classes
# Model: ST-GCN 
# KFOLD = 5, Validation and Test are the same
# Filterout hardcases
class Entrypoint(TrainEntrypoint[IN, OUT, BaseConfig]):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=False),
                filterout_hardcases=True,
                load_dir="./Data/output_2.pkl",
                savename="processed_unlabeled_subset.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=72,
            try_name="stgcn-12classes-prediction",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    @property
    def criteria_names(self) -> List[str]:
        return super().criteria_names + ['ACC']
    
    @property
    def best_epoch_criterion_idx(self) -> int:
        return self.criteria_names.index('ACC')
       
    def get_model(self):
        num_classes = self.kfold._ulabels.size
        model = Model(2, num_classes, True, None)
        return model
        
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0]    # B=batch size, T, V, C=3
        x = x[..., [0,1]]  # B=batch size, T, V, C=2
        x = torch.permute(x, (0, 3, 1, 2)) # B=batch size, C=2, T, V
        x = torch.unsqueeze(x, dim=-1)
        # N=b, C=in_channel, T=length of input sequence, V=number of graph nodes, M=number of instance in a frame
        x = self.model(x)
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        y = data[1]
        idx = torch.arange(y.numel())
        m = torch.nn.LogSoftmax(dim=1)
        output = m(x)
        loss = -torch.mean(output[idx, y]) #CE
        return loss

    def run():
        model = self.get_model()
        model.load_state_dict(torch.load('/home/lisa/STP-Gait/results/71_stgcn-12classes/weights/0-0/102'))
        model.to(self.conf.device)



        




