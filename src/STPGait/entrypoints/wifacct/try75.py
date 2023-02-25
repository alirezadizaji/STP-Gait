import torch
from torch import nn

from ...config import BaseConfig, TrainingConfig
from ...dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ...data.read_gait_data import ProcessingGaitConfig
from ...enums import Optim
from ...models.wifacct import WiFaCCT
from ...models.wifacct.vivit import Model1, Model2
from ...preprocess.main import PreprocessingConfig
from ..train import TrainEntrypoint
from .try63 import Entrypoint as E


# try 75
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=False),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=75,
            try_name="wifacct_vivit",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50, batch_size=32)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    @property
    def frame_size(self):
        return 400
    
    @property
    def chunk_size(self):
        return 10

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size

        d = 80
        model1 = Model1(d_model=d, nhead=8, n_enc_layers=2)
        model2 = Model2(num_classes, d_model=d, nhead=8, n_enc_layers=1)
        
        model = WiFaCCT[Model1, Model2](model1, model2, num_aux_branches=3)
        return model

    def _model_forwarding(self, data):
        x = data[0]
        x = x[:, :self.frame_size, :, [0, 1]]          # B, T, V, C
        x = torch.stack(x.split(self.chunk_size, 1), dim=3) # B, T1, V, T2, C
        x = x.flatten(3).to(self.conf.device)  # B, T1, V, D

        out = self.model(x, m1_args=dict(), m2_args=dict())
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

        loss = 0.2 * loss_unsup
        if not torch.isnan(loss_sup):
            loss = loss + loss_sup

        return loss