import os
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


from ..context.skeleton import Skeleton
from ..data import proc_gait_data
from ..enums import EdgeType
from ..preprocess.pad_empty_frames import pad_empty_frames

class SkeletonDataset(Dataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray,
            edge_type: EdgeType = EdgeType.DEFAULT,
            use_lower_part: bool = False,
            **kwargs) -> None:

        super().__init__()
        lower_part_node_indices = np.delete(np.arange(Skeleton.num_nodes), Skeleton.upper_part_node_indices)
        
        if use_lower_part:
            X = X[..., lower_part_node_indices, :]

        X: torch.Tensor = torch.from_numpy(X).float()
        Y: torch.Tensor = torch.from_numpy(Y)

        assert X.size(0) == names.size == Y.size(0), f"Mismatch number of samples between data ({X.size(0)}), names ({names.size}) and labels ({Y.size(0)})."
        N, T, V, C = X.size()

        # Take edge index set
        if edge_type == EdgeType.DEFAULT:
            edge_index = torch.from_numpy(Skeleton.get_vanilla_edges(T, use_lower_part)[0])
        elif edge_type == EdgeType.INTER_FRAME_M1:
            edge_index = Skeleton.get_simple_interframe_edges(T, use_lower_part=use_lower_part, **kwargs)
        elif edge_type == EdgeType.INTER_FRAME_M2:
            edge_index = Skeleton.get_interframe_edges_mode2(T, use_lower_part=use_lower_part, **kwargs)
        
        X = X.reshape(N, T * V, C)
        self.data: List[Data] = [Data(x=x, edge_index=edge_index, y=y, name=n) 
            for x, n, y in zip(X, names, Y)]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Batch:
        if not isinstance(index, list):
            index = [index]
        return [self.data[i] for i in index]