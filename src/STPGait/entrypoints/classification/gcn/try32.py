import torch

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Label, Optim
from ....preprocess.main import PreprocessingConfig
from .try28 import Entrypoint as E
from ...train import TrainEntrypoint

# try 32 (try 28 ->)
## Use anxious-hypokinetic_frontal-healthy labels
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, remove_labels=[Label.ATAXIC, Label.PARETIC, Label.SENSORY_ATAXIC]),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                filterout_hardcases=True,
                savename="processed_120c_xyz.pkl",
                proc_conf=ProcessingGaitConfig(fillZ_empty=False, preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=28,
            try_name="gcn_non_temporal",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

        self._edge_index: torch.Tensor = None

    def _get_edges(self, num_frames: int):
        return torch.from_numpy(Skeleton.get_vanilla_edges(num_frames)[0])
