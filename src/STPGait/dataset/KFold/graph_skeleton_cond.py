from typing import Dict

import numpy as np

from .skeleton_cond import SkeletonCondKFoldConfig, SkeletonCondKFoldOperator
from ..graph_skeleton_cond import GraphSkeletonCondDataset
from ...enums import Separation

class GraphSkeletonCondKFoldOperator(SkeletonCondKFoldOperator[GraphSkeletonCondDataset, SkeletonCondKFoldConfig]):
    r""" KFold for Graph Skeleton Dataset """

    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, GraphSkeletonCondDataset]:

        sets: Dict[Separation, GraphSkeletonCondDataset] = dict()
        labeled = self.get_labeled_mask()

        sets[Separation.VAL] = GraphSkeletonCondDataset(self._x[val_indices], self._names[val_indices], 
            self._numeric_labels[val_indices], labeled=labeled[val_indices], 
            node_invalid=self._node_invalidity[val_indices], cond_mask=self._cond_mask[val_indices])
        
        sets[Separation.TEST] = GraphSkeletonCondDataset(self._x[test_indices], self._names[test_indices], 
            self._numeric_labels[test_indices], labeled=labeled[test_indices], 
            node_invalid=self._node_invalidity[test_indices], cond_mask=self._cond_mask[test_indices])
        
        sets[Separation.TRAIN] = GraphSkeletonCondDataset(self._x[train_indices], self._names[train_indices], 
            self._numeric_labels[train_indices], labeled=labeled[train_indices], 
            node_invalid=self._node_invalidity[train_indices], cond_mask=self._cond_mask[train_indices])
        
        return sets
