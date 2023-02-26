from dataclasses import dataclass
import os
from typing import Dict, Generic, TypeVar

import numpy as np
import pandas as pd

from .core import KFoldOperator, KFoldConfig
from .skeleton import SkeletonKFoldOperator
from ...data import proc_gait_data_v2
from ..skeleton import SkeletonDataset
from ...enums import Separation
from ...context import Skeleton
from ...preprocess.pad_empty_frames import pad_empty_frames
from ...data.read_gait_data import ProcessingGaitConfig

@dataclass
class SkeletonKFoldConfig:
    kfold_config: KFoldConfig
    load_dir: str
    savename: str = "processed.pkl"
    proc_conf: ProcessingGaitConfig = ProcessingGaitConfig()
    filterout_hardcases: bool = False

class SkeletonCondKFoldOperator(SkeletonKFoldOperator):
    r""" KFold for Skeleton Dataset condition

    Args:
        config (KFoldSkeletonConfig, optional): Configuration for the Skeleton KFold operator
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train/test/validation SkeletonDataset instances.
    """

    def __init__(self, config) -> None:
        self.conf = config
        assert os.path.isfile(self.conf.load_dir), f"The given directory ({self.conf.load_dir}) should determine a file path (CSV); got directory instead."
        root_dir = os.path.dirname(self.conf.load_dir)
        save_dir = os.path.join(root_dir, self.conf.savename)
        if not os.path.exists(save_dir):
            proc_gait_data_v2(self.conf.load_dir, root_dir, self.conf.savename, self.conf.proc_conf)
        
        with open(save_dir, "rb") as f:
            import pickle
            x, labels, names, hard_cases_id = pickle.load(f)
        
        print(f"### SkeletonKFoldOperator (raw data info): x shape {x.shape} ###", flush=True)
        
        values, counts = np.unique(labels, return_counts=True)
        v_to_c = {v: c for v, c in zip(values, counts)}
        print(f"### Labels counts (before pruning): {v_to_c} ###", flush=True)

        with open(self.conf.load_dir, "rb") as f:
            df = pd.read_pickle(f)
    
        condition = df['condition'].values
        id = df['id'].values
        
        if self.conf.filterout_hardcases:
            mask = np.ones(x.shape[0], dtype=np.bool)
            mask[hard_cases_id] = False
            x = x[mask]
            labels = labels[mask]
            names = names[mask]
            condition = condition[mask]
            id = id[mask]
            print(f"### SkeletonKFoldOperator (Filtering hard cases): x shape {x.shape} ###", flush=True)

        idu, id_uv = np.unique(id, return_index=True)
        condu, cond_uv = np.unique(condition, return_index=True)
        
        N = idu.size
        M = condu.size
        
        new_x = np.full((N, M, *x.shape[1:]), fill_value=-np.inf)        # N, M, T, V, C
        new_labels = np.full((N, M), fill_value=-np.inf)                    # N, M

        for x1, y, id, cond in zip(x, labels, id_uv, cond_uv):
            new_x[id, cond] = x1
            new_labels[id, cond] = y

        # All labels of a patient in different condition must be same
        assert np.all(new_labels[id, [0]] == new_labels), "There is a patient with at least two different labels."
        
        labels = new_labels[:, 0]                                         # N
        x = new_x
        values, counts = np.unique(labels, return_counts=True)
        v_to_c = {v: c for v, c in zip(values, counts)}
        print(f"### Labels counts (after pruning): {v_to_c} ###", flush=True)

        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._x = x[indices]
        self._labels = labels[indices]
        self._names = names[indices]
        self._hard_cases_id = hard_cases_id
        self._node_invalidity = self._get_node_invalidity()         # ..., V
        self._frame_invalidity = self._node_invalidity.sum(-1) > 0

        super().__init__(**self.conf.kfold_config.__dict__)

    def _get_node_invalidity(self) -> np.ndarray:
        """ It generates a mask, determining whether a node within a frame is missed or not. It verifies
        this by measuring the ratio distance of joints with respect to the center, heap, and knee 
        joints both vertically and horizontally.

        Returns:
            np.ndarray: A mask, If an element is True, then that frame on that sequence has some missed information.
        """
        N, M, T, *_ = self._x.shape
        mask = np.zeros((N, M, T), dtype=np.bool)

        # First, mask frames having at least one NaN values
        nan_mask = np.isnan(self._x)
        idx1, idx2, idx3, idx4, _ = np.nonzero(nan_mask)
        mask[idx1, idx2, idx3] = True
        ## Replace center location with NaN joints
        self._x[idx1, idx2, idx3, idx4, :] = self._x[idx1, idx2, idx3, [Skeleton.CENTER], :] 
        assert np.all(~np.isnan(self._x)), "There is still NaN values among locations."

        # Second, checkout non NaN frames if there are some inconsistency between shoulders, toes, or heads locations.
        node_invalidity = np.zeros_like(self._x[..., 0], dtype=np.bool)
        
        ## Critieria joints
        center = self._x[..., [Skeleton.CENTER], :] # N, M, T, 1, C
        lheap = self._x[..., [Skeleton.LEFT_HEAP], :] # N, M, T, 1, C
        lknee = self._x[..., [Skeleton.LEFT_KNEE], :] # N, M, T, 1, C

        ## Upper body verification
        upper_indices = np.concatenate([Skeleton.SHOULDERS, Skeleton.HEAD_EYES_EARS])
        y_upper = self._x[..., upper_indices, 1] # N, M, T, VV
        y_center = center[..., 1]
        y_lknee = lknee[..., 1]
        mask1 = ((y_upper - y_center) / (y_upper - y_lknee + 1e-3)) < 0.1
        node_invalidity[..., upper_indices] = mask1
        
        ## Lower body verification
        lower_indices = Skeleton.WHOLE_FOOT
        y_lower = self._x[..., lower_indices, 1] # N, T, VVV
        mask2 = ((y_center - y_lower) / (y_center - y_lknee + 1e-3)) < 1
        node_invalidity[..., lower_indices] = mask2
        
        ## Horizontal verification
        horizon_indices = Skeleton.ELBOWS_HANDS
        x_horizon = np.abs(self._x[:, :, horizon_indices, 0]) # N, T, VVVV
        x_center = center[..., 0]
        x_lheap = lheap[..., 0]
        mask3 = ((x_horizon - x_center) / (x_lheap - x_center + 1e-3)) < 0.1 # N, T, VV
        node_invalidity[:, :, horizon_indices] = mask3

        return node_invalidity
        
        
    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, SkeletonDataset]:
        
        sets: Dict[Separation, SkeletonDataset] = dict()
        labeled = self.get_labeled_mask()

        sets[Separation.VAL] = SkeletonDataset(self._x[val_indices], self._names[val_indices], 
            self._numeric_labels[val_indices], labeled[val_indices], 
            self._frame_invalidity[val_indices])
        
        sets[Separation.TEST] = SkeletonDataset(self._x[test_indices], self._names[test_indices], 
            self._numeric_labels[test_indices], labeled[test_indices], 
            self._frame_invalidity[test_indices])
        
        
        sets[Separation.TRAIN] = SkeletonDataset(self._x[train_indices], self._names[train_indices], 
                self._numeric_labels[train_indices], labeled[train_indices], 
                self._frame_invalidity[train_indices])
        
        return sets
