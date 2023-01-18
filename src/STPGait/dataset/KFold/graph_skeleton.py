from dataclasses import dataclass
import os
from typing import Callable, Dict

import numpy as np
import torch

from .core import KFoldOperator, KFoldConfig
from .skeleton import SkeletonKFoldConfig, SkeletonKFoldOperator
from ...data import proc_gait_data
from ..graph_skeleton import GraphSkeletonDataset
from ...enums import Separation
from ...context import Skeleton
from ...preprocess.pad_empty_frames import pad_empty_frames
from ...data.read_gait_data import ProcessingGaitConfig

@dataclass
class GraphSkeletonKFoldConfig(SkeletonKFoldConfig):
    get_edge_index: Callable[[int], 'torch.Tensor'] = lambda T: torch.from_numpy(Skeleton.get_interframe_edges_mode2(T, I=30, offset=20))

class GraphSkeletonKFoldOperator(SkeletonKFoldOperator[GraphSkeletonDataset, GraphSkeletonKFoldConfig]):
    r""" KFold for Graph Skeleton Dataset """

    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, GraphSkeletonDataset]:
        sets: Dict[Separation, GraphSkeletonDataset] = dict()
        sets[Separation.VAL] = GraphSkeletonDataset(self._x[val_indices], self._names[val_indices], 
            self._label_indices[val_indices], self.conf.get_edge_index)
        sets[Separation.TEST] = GraphSkeletonDataset(self._x[test_indices], self._names[test_indices], 
            self._label_indices[test_indices], self.conf.get_edge_index)
        
        # Train dataset should not have missed frames at all.
        train_not_missed = ~self._mask[train_indices]
        x = self._x[train_indices]
        new_x = np.zeros_like(x)
        seq_i, frame_i = np.nonzero(train_not_missed)
        _, counts = np.unique(seq_i, return_counts=True)
        frame_ii = np.concatenate([np.arange(c) for c in counts]).flatten()
        new_x[seq_i, frame_ii] = x[seq_i, frame_i]
        
        self._x[train_indices] = pad_empty_frames(new_x)
        sets[Separation.TRAIN] = GraphSkeletonDataset(self._x[train_indices], self._names[train_indices], 
                self._label_indices[train_indices], self.conf.get_edge_index)
        
        return sets
