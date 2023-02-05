import torch
from torch_geometric.data import Batch, Data

from ....config import BaseConfig, TrainingConfig
from ....context import Skeleton
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.gcn_semisupervised import calc_edge_weight
from ....preprocess.main import PreprocessingConfig
from .try18 import Entrypoint as E
from ...train import TrainEntrypoint

# try 40 (try 18 ->)
## Zero out edge weights whose one end have at least one invalid node.
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c.pkl",
                proc_conf=ProcessingGaitConfig(preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=40,
            try_name="gcn3l_non_temporal",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def _model_forwarding(self, data):
        x = data[0][..., [0, 1]].to(self.conf.device) # Use X-Y features
        node_invalid = data[2].flatten(1).to(x.device)
        node_valid = ~node_invalid

        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1))
        x = x.flatten(1, -2) # N, T*V, D
        data = Batch.from_data_list([Data(x=x_, edge_index=self._edge_index, edge_weight=calc_edge_weight(self._edge_index, nv)) for (x_, nv) in zip(x, node_valid)])
        data = data.to(x.device)
        out = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_weight=data.edge_weight)
        return out