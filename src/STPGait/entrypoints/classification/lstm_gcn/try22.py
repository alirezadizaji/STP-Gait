from typing import Tuple

import torch

from ....config import BaseConfig, TrainingConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....enums import Optim, Label
from ....models import GCNLSTMTransformerV2
from ....preprocess.main import PreprocessingConfig
from ..lstm_gcn_transformer.try7 import Entrypoint as E
from ..lstm_gcn_transformer.try2 import Entrypoint as EE

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
OUT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

# try 22 (try 7 ->)
## Remove transformer and cnn part
## Filterout hard cases
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ATAXIC, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=22,
            try_name="lstm_gcn",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super(EE, self).__init__(kfold, config)

    def get_model(self):
        return GCNLSTMTransformerV2(cnn_conf=None, transformer_encoder_conf=None)