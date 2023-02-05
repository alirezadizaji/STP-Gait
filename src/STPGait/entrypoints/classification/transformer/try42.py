from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim
from ....models import Transformer
from ....preprocess.main import PreprocessingConfig
from ..lstm_gcn_transformer.try4 import Entrypoint as E
from ...train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 42 (try 4 ->)
## Use transformer model.
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0),
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=42,
            try_name="transformer",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)


    def get_model(self):
        return Transformer()
    
    def _model_forwarding(self, data: IN) -> OUT:
        x, y = data
        x = x[..., [0, 1]].flatten(2)
        x = self.model(x.to(self.conf.device), y.to(self.conf.device))
        return x