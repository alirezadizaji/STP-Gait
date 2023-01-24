from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....preprocess.main import PreprocessingConfig
from .try14 import Entrypoint as E

# try 15 (try 14 ->)
## only using the lower part of the skeleton sequence. 
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=10, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=15,
            try_name="gcn3l_m1_lower_part_dil_30",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super(E, self).__init__(kfold, config)

    def _get_edges(self, num_frames: int):
        return Skeleton.get_simple_interframe_edges(num_frames, dilation=30, use_lower_part=True)