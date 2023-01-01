from tqdm import tqdm

from dig.xgraph.models import GCN_3l_BN
from torch.utils.data import DataLoader
import torch

from .context import Skeleton
from .dataset import DatasetInitializer

def run():
    initializer = DatasetInitializer("../../Data/")

    model = GCN_3l_BN(model_level='graph', dim_node=2, dim_hidden=30, num_classes=12)
    edges = None
    with initializer:
        train_loader = DataLoader(initializer.train, batch_size=32, shuffle=True)
        val_loader = DataLoader(initializer.val, batch_size=32)
        val_loader = DataLoader(initializer.test, batch_size=32)

        for data, labels, _ in tqdm(train_loader):
            # Ignore Z dimension
            data: torch.Tensor = data[..., [0, 1]]    # N, T, V, 2
            data = data.view(N, T * V, C)
            N, T, V, C = data.size()
            if edges is None:
                edges = Skeleton.get_simple_interframe_edges(T)

            output = model(x=data, edge_index=edges)
