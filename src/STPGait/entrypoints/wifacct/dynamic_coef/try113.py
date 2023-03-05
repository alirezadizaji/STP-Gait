from typing import List, Tuple

import torch
from torch import nn

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.wifacct import WiFaCCT
from ....models.wifacct.lstm_gcn import Model
from ....preprocess.main import PreprocessingConfig
from ...train import TrainEntrypoint
from ..try63 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor]

# try 113 (63 ->)
## Semi-supervised training using LSTM-GCN basic blocks
## Dynamic coefficient loss
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
            try_num=113,
            try_name="wifacct_lstm_gcn",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=32)
        )
        TrainEntrypoint.__init__(self, kfold, config)
        self._start_ul_epoch: int = 10


    def get_model(self) -> nn.Module:
        model1 = Model(n=1, get_gcn_edges= lambda T: torch.from_numpy(Skeleton.get_vanilla_edges(T)[0]))
        model2 = Model(n=1, get_gcn_edges= lambda T: torch.from_numpy(Skeleton.get_vanilla_edges(T)[0]))
        
        model = WiFaCCT[Model, Model](model1, model2, num_aux_branches=3)
        return model

    def _model_forwarding(self, data):
        x, _, node_invalid, _ = data[0][..., [0, 1]].to(self.conf.device) # Use X-Y features
        node_valid = ~node_invalid.flatten(1)

        out = self.model(x, m1_args=dict(x_valid=node_valid), m2_args=dict(x_valid=node_valid))
        return out

    def _calc_loss(self, x, data):
        _, y, _, labeled = data
        o_main, o_aux = x
        
        oml = o_main[labeled]
        yl = y[labeled]
        loss_sup = -torch.mean(oml[torch.arange(yl.numel()), yl])

        y1d = o_main.argmax(1).detach().unsqueeze(1).repeat(1, o_aux.size(1)).flatten()
        o_aux = o_aux.flatten(0, 1)
        loss_unsup = -torch.mean(o_aux[torch.arange(y1d.size(0)), y1d])

        u = int(self.epoch >= self._start_ul_epoch or not self.model.training)
        if not torch.isnan(loss_sup):
            alpha = loss_sup.detach() / loss_unsup.detach()
            loss = loss_sup + u * alpha * loss_unsup
        else:
            loss = loss_unsup
        return loss