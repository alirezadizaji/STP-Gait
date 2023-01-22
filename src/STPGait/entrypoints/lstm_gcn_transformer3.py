from typing import Tuple

import torch

from ..config import BaseConfig, TrainingConfig
from ..data.read_gait_data import ProcessingGaitConfig
from ..dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ..enums import Optim
from ..preprocess.main import PreprocessingConfig
from .lstm_gcn_transformer2 import Entrypoint as E

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# I3 (I2 ->)
## Edge attributes added: 0 -> has at least one invalid node, 1 -> both nodes are valid.
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=4,
            try_name="lstm_gcn_transformer",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super(E, self).__init__(kfold, config)


    def _model_forwarding(self, data: IN) -> OUT:
        x, y, node_invalid = data
        x = x[..., [0, 1]].flatten(2)
        node_valid = ~node_invalid.flatten(1)
        x = self.model(x.to(self.conf.device), y.to(self.conf.device), node_valid.to(self.conf.device))
        return x