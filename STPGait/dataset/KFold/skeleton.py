import os
from typing import Dict

import numpy as np

from .kfold import KFoldInitializer
from ...data import proc_gait_data
from ..skeleton import SkeletonDataset
from ...enums import EdgeType, Separation
from ...preprocess.pad_empty_frames import pad_empty_frames

class KFoldSkeleton(KFoldInitializer[SkeletonDataset]):
    r""" KFold for Skeleton Dataset

    Args:
        load_dir (str): Where to load raw gait data; pickle format
        fillZ_empty (bool, optional): If `True`, then fill Z with zero value, O.W. process patient steps to get approximate Z locations. Defaults to `True`.
        filterout_unlabeled (bool, optional): If `True`, then filter out cases labeled as `unlabeled`. Defaults to `True`.
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train/test/validation SkeletonDataset instances.
    """
    def __init__(self, 
            K: int,
            init_valK: int,
            init_testK: int,
            load_dir: str,
            fillZ_empty: bool = True,
            filterout_unlabeled: bool = True) -> None:

        root_dir = os.path.dirname(load_dir)
        save_dir = os.path.join(root_dir, "processed.pkl")
        if not os.path.exists(save_dir):
            proc_gait_data(load_dir, root_dir, fillZ_empty)
        
        with open(save_dir, "rb") as f:
            import pickle
            x, labels, names, missed_head = pickle.load(f)
        
        if filterout_unlabeled:
            mask = labels != 'unlabeled'
            x = x[mask]
            labels = labels[mask]
            names = names[mask]
            missed_head = missed_head[mask]
        
        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._x = x[indices]
        self._labels = labels[indices]
        self._names = names[indices]
        self._missed_head = missed_head[indices]

        super().__init__(K, init_valK, init_testK)

    def get_labels(self) -> np.ndarray:
        return self._labels

    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, SkeletonDataset]:
        sets: Dict[Separation, SkeletonDataset] = dict()

        set[Separation.TRAIN] = SkeletonDataset(self._x[train_indices], self._names[train_indices], self._label_indices[train_indices])
        set[Separation.VAL] = SkeletonDataset(self._x[val_indices], self._names[val_indices], self._label_indices[val_indices])
        set[Separation.TEST] = SkeletonDataset(self._x[test_indices], self._names[test_indices], self._label_indices[test_indices])
        
        return sets
