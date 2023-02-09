from typing import List

from .....config import BaseConfig, TrainingConfig
from .....data.read_gait_data import ProcessingGaitConfig
from .....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from .....enums import Optim
from .....models.others.vivit import ViViT, Encoder4
from .....preprocess.main import PreprocessingConfig
from ....train import TrainEntrypoint
from .try45 import Entrypoint as E

# Try 48 (45 ->)
# Model: ViViT (Encoder4)
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                filterout_hardcases=True,
                load_dir="../../Data/output_1.pkl",
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=48,
            try_name="vivit_m4_t40_f400",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

    def get_model(self):
        encoder = Encoder4(80, 2000, 800, 8, 3)
        num_classes = self.kfold._ulabels.size
        model = ViViT(35, num_classes, encoder)
        return model