from typing import Dict

import numpy as np

from .skeleton import SkeletonKFoldConfig, SkeletonKFoldOperator
from ..graph_skeleton import GraphSkeletonDataset
from ...enums import Separation

class GraphSkeletonKFoldOperator(SkeletonKFoldOperator[SkeletonKFoldConfig, SkeletonKFoldConfig]):
    r""" KFold for Graph Skeleton Dataset """

    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, GraphSkeletonDataset]:
        sets: Dict[Separation, GraphSkeletonDataset] = dict()
        sets[Separation.VAL] = GraphSkeletonDataset(self._x[val_indices], self._names[val_indices], 
            self._label_indices[val_indices], node_invalid=self._node_invalidity[val_indices])
        sets[Separation.TEST] = GraphSkeletonDataset(self._x[test_indices], self._names[test_indices], 
            self._label_indices[test_indices], node_invalid=self._node_invalidity[test_indices])
        sets[Separation.TRAIN] = GraphSkeletonDataset(self._x[train_indices], self._names[train_indices], 
            self._label_indices[train_indices], node_invalid=self._node_invalidity[train_indices])
        
        return sets
