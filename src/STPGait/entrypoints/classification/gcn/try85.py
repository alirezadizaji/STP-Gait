import torch

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonCondKFoldOperator, SkeletonCondKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Condition, Optim
from ....preprocess.main import PreprocessingConfig
from .try82 import Entrypoint as E
from ...train import TrainEntrypoint

# try 85 (try 82 ->)
## Use only DTC Condition
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonCondKFoldOperator(
            config=SkeletonCondKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0, filterout_unlabeled=True),
                load_dir="../../Data/cond12class.pkl",
                filterout_hardcases=True,
                savename="processed12cls_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)),
                condition=Condition.DTC)
            )
        config = BaseConfig(
            try_num=85,
            try_name="gcn3l_non_temporal",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def _get_edges(self, num_frames: int):
        return torch.from_numpy(Skeleton.get_vanilla_edges(num_frames)[0])

    def data_preprocessing(self, data):
        data[0] = data[0].squeeze(1)
        assert data[0].ndim == 4
        return data