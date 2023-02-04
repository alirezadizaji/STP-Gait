import torch
from torch import nn
from torch_geometric.data import Data, Batch

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models.gcn_3l_bn import GCN_3l_BN
from ....preprocess.main import PreprocessingConfig
from .try17 import Entrypoint as E
from ...train import TrainEntrypoint

# try 27 (try 17 ->)
## Approximate and use Z feature too.
class Entrypoint(E):
    def __init__(self) -> None:
        kfold = GraphSkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=5, init_valK=0, init_testK=0),
                load_dir="../../Data/output_1.pkl",
                filterout_hardcases=True,
                savename="processed_120c_xyz.pkl",
                proc_conf=ProcessingGaitConfig(fillZ_empty=False, preprocessing_conf=PreprocessingConfig(critical_limit=120)))
            )
        config = BaseConfig(
            try_num=27,
            try_name="gcn3l_m2_I_60_offset_30",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def _model_forwarding(self, data):
        x = data[0]

        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1))
        x = x.flatten(1, -2) # N, T*V, D
        data = Batch.from_data_list([Data(x=x_, edge_index=self._edge_index) for x_ in x])
        data = data.to(x.device)
        out = self.model(x=data.x, edge_index=data.edge_index, batch=data.batch)
        return out

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size
        model = GCN_3l_BN(model_level='graph', dim_node=3, dim_hidden=60, num_classes=num_classes)
        return model
