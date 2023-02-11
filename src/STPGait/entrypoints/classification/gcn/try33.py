from typing import List

import torch
from torch import nn
from torch_geometric.data import Data, Batch

from ....config import BaseConfig, TrainingConfig
from ....dataset.KFold import GraphSkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ....data.read_gait_data import ProcessingGaitConfig
from ....enums import Optim
from ....models import GCNTransformer
from ....preprocess.main import PreprocessingConfig
from .try17 import Entrypoint as E
from ...train import TrainEntrypoint

# try 33 (try 17 ->)
## Use GCN-Transformer model
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
            try_num=33,
            try_name="gcn_transformer_m2_I_60_offset_30",
            device="cuda:0",
            eval_batch_size=32,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        TrainEntrypoint.__init__(self, kfold, config)

        self._edge_index: torch.Tensor = None

    def get_model(self) -> nn.Module:
        num_classes = self.kfold._ulabels.size
        return GCNTransformer(model_level="graph", dim_node=2, num_nodes=25, dim_hidden=60, 
            num_classes=num_classes)

    def _model_forwarding(self, data):
        x = data[0][..., [0, 1]].to(self.conf.device) # Use X-Y features
        if self._edge_index is None:
            self._edge_index = self._get_edges(x.size(1)).to(x.device)

        x = x.flatten(1, -2) # N, T*V, D
        batch_size = x.size(0)

        data = Batch.from_data_list([Data(x=x_, edge_index=self._edge_index) for x_ in x])
        data = data.to(x.device)
        out = self.model(batch_size, x=data.x, edge_index=data.edge_index, batch=data.batch)
        return out