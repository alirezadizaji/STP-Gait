from dataclasses import dataclass
import os
from typing import Dict, Generic, TypeVar

import numpy as np

from .core import KFoldOperator, KFoldConfig
from ...data import proc_gait_data
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
    proc_conf: ProcessingGaitConfig = ProcessingGaitConfig(),
    filterout_unlabeled: bool = True

TT = TypeVar('TT', bound=SkeletonDataset)
C = TypeVar('C', bound=SkeletonKFoldConfig)
class SkeletonKFoldOperator(KFoldOperator[TT], Generic[TT, C]):
    r""" KFold for Skeleton Dataset

    Args:
        config (KFoldSkeletonConfig, optional): Configuration for the Skeleton KFold operator
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train/test/validation SkeletonDataset instances.
    """
    def __init__(self, config: C) -> None:
        self.conf: C = config
        assert os.path.isfile(self.conf.load_dir), f"The given directory ({self.conf.load_dir}) should determine a file path (CSV); got directory instead."
        root_dir = os.path.dirname(self.conf.load_dir)
        save_dir = os.path.join(root_dir, self.conf.savename)
        if not os.path.exists(save_dir):
            proc_gait_data(self.conf.load_dir, root_dir, self.conf.savename, self.conf.proc_conf)
        
        with open(save_dir, "rb") as f:
            import pickle
            x, labels, names, hard_cases_id = pickle.load(f)
        
        if self.conf.filterout_unlabeled:
            mask = labels != 'unlabeled'
            x = x[mask]
            labels = labels[mask]
            names = names[mask]
            hard_cases_id = hard_cases_id[mask]
        
        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._x = x[indices]
        self._labels = labels[indices]
        self._names = names[indices]
        self._hard_cases_id = hard_cases_id
        self._mask = self._generate_missed_frames()

        super().__init__(self.conf.kfold_config.K, self.conf.kfold_config.init_valK, self.conf.kfold_config.init_testK)

    def _generate_missed_frames(self) -> np.ndarray:
        """ It generates a mask, determining whether a frame is missed or not. It verifies
        this by measuring the ratio distance of joints with respect to the center, heap, and knee 
        joints both vertically and horizontally.

        Returns:
            np.ndarray: A mask, If an element is True, then that frame on that sequence has some missed information.
        """
        N, T = self._x.shape[0], self._x.shape[1]
        mask = np.zeros((N, T), dtype=np.bool)

        # First, mask frames having at least one NaN values
        nan_mask = np.isnan(self._x)
        idx1, idx2, idx3, _ = np.nonzero(nan_mask)
        mask[idx1, idx2] = True
        ## Replace center location with NaN joints
        self._x[idx1, idx2, idx3, :] = self._x[idx1, idx2, [Skeleton.CENTER], :] 
        assert np.all(~np.isnan(self._x)), "There is still NaN values among locations."

        # Second, checkout non NaN frames if there are some inconsistency between shoulders, toes, or heads locations.
        ## Critieria joints
        x_center = self._x[:, :, [Skeleton.CENTER], :] # N, T, 1, C
        x_lheap = self._x[:, :, [Skeleton.LEFT_HEAP], :] # N, T, 1, C
        x_lknee = self._x[:, :, [Skeleton.LEFT_KNEE], :] # N, T, 1, C

        ## Upper body
        x_elbows_hands = np.abs(self._x[:, :, Skeleton.ELBOWS_HANDS, :]) # N, T, V1, C
        x_shoulders = self._x[:, :, Skeleton.SHOULDERS, :] # N, T, V2, C
        x_heads_eyes_ears = self._x[:, :, Skeleton.HEAD_EYES_EARS, :] # N, T, V3, C
        
        ## Lower body
        x_foots = self._x[:, :, Skeleton.WHOLE_FOOT, :] # N, T, V4, C

        ## Verifications
        vertical_up = np.concatenate((x_shoulders, x_heads_eyes_ears), axis=2)[..., 1] # N, T, VV
        mask1 = ((vertical_up - x_center[..., 1]) / (vertical_up - x_lknee[..., 1] + 1e-3)) < 0.1
        mask1 = mask1.sum(2) > 0
        
        vertical_down = x_foots[..., 1] # N, T, VVV
        mask2 = ((x_center[..., 1] - vertical_down) / (x_center[..., 1] - x_lknee[..., 1] + 1e-3)) < 1
        mask2 = mask2.sum(2) > 0

        horizontal = x_elbows_hands[..., 0] # N, T, VVVV        
        mask3 = ((horizontal - x_center[..., 0]) / (x_lheap[..., 0] - x_center[..., 0] + 1e-3)) < 0.1 # N, T, VV
        mask3 = mask3.sum(2) > 0

        mask = np.logical_or(np.logical_or(mask, mask1), np.logical_or(mask2, mask3))
        return mask
        
        
    def get_labels(self) -> np.ndarray:
        return self._labels

    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, SkeletonDataset]:
        sets: Dict[Separation, SkeletonDataset] = dict()
        sets[Separation.VAL] = SkeletonDataset(self._x[val_indices], self._names[val_indices], 
            self._label_indices[val_indices], self._mask[val_indices])
        sets[Separation.TEST] = SkeletonDataset(self._x[test_indices], self._names[test_indices], 
            self._label_indices[test_indices], self._mask[test_indices])
        
        # Train dataset should not have missed frames at all.
        train_not_missed = ~self._mask[train_indices]
        x = self._x[train_indices]
        new_x = np.zeros_like(x)
        seq_i, frame_i = np.nonzero(train_not_missed)
        _, counts = np.unique(seq_i, return_counts=True)
        frame_ii = np.concatenate([np.arange(c) for c in counts]).flatten()
        new_x[seq_i, frame_ii] = x[seq_i, frame_i]
        
        self._x[train_indices] = pad_empty_frames(new_x)
        sets[Separation.TRAIN] = SkeletonDataset(self._x[train_indices], self._names[train_indices], 
                self._label_indices[train_indices])
        
        return sets
