from dataclasses import dataclass
import os
from typing import Dict, Generic, List, TypeVar

import numpy as np

from .core import KFoldOperator, KFoldConfig
from ...data import read_multi_pkl_gait_data
from ..skeleton import SkeletonDataset
from ...enums import Separation
from ...context import Skeleton
from ...preprocess.pad_empty_frames import pad_empty_frames
from ...data.read_gait_data import ProcessingGaitConfig
from .skeleton import SkeletonKFoldConfig, SkeletonKFoldOperator

@dataclass
class SkeletonKFoldMultiPklConfig(SkeletonKFoldConfig):
    load_dir: List[str] = []
    unite_save_dir: str = None

TT = TypeVar('TT', bound=SkeletonDataset)
C = TypeVar('C', bound=SkeletonKFoldMultiPklConfig)
class SkeletonMultiPklKFoldOperator(SkeletonKFoldOperator[TT, C]):
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
        prev_K = self.conf.kfold_config.K
        self.conf.kfold_config.K = len(self.conf.load_dir)
        print(f"@@@ SkeletonMultiPklKFoldOperator WARNING: KFold is changed from {prev_K} to {self.conf.kfold_config.K} permanently. @@@", flush=True)

        assert os.path.isfile(self.conf.load_dir), f"The given directory ({self.conf.load_dir}) should determine a file path (CSV); got directory instead."
        root_dir = os.path.dirname(self.conf.load_dir)
        save_dir = os.path.join(root_dir, self.conf.savename)
        if not os.path.exists(save_dir):
            read_multi_pkl_gait_data(self.conf.load_dir, self.conf.unite_save_dir, root_dir, self.conf.savename, self.conf.proc_conf)
        
        with open(save_dir, "rb") as f:
            import pickle
            x, labels, names, hard_cases_id, pkl_no = pickle.load(f)
        
        print(f"### SkeletonMultiPklKFoldOperator (raw data info): x shape {x.shape} ###", flush=True)

        values, counts = np.unique(labels, return_counts=True)
        v_to_c = {v: c for v, c in zip(values, counts)}
        print(f"### Labels counts (before pruning): {v_to_c} ###", flush=True)

        if self.conf.filterout_hardcases:
            mask = np.ones(x.shape[0], dtype=np.bool)
            mask[hard_cases_id] = False
            x = x[mask]
            labels = labels[mask]
            names = names[mask]
            pkl_no = pkl_no[mask]
            print(f"### SkeletonMultiPklKFoldOperator (Filtering hard cases): x shape {x.shape} ###", flush=True)

        values, counts = np.unique(labels, return_counts=True)
        v_to_c = {v: c for v, c in zip(values, counts)}
        print(f"### Labels counts (after pruning): {v_to_c} ###", flush=True)

        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._x = x[indices]
        self._labels = labels[indices]
        self._names = names[indices]
        self._pkl_no = pkl_no[indices]
        self._node_invalidity = self._get_node_invalidity()         # ..., V
        self._frame_invalidity = self._node_invalidity.sum(-1) > 0

        super().__init__(**self.conf.kfold_config.__dict__)

    def split_labeled(self) -> List[np.ndarray]:
        labels = self.get_labels()

        label_to_splits: Dict[int, List[np.ndarray]] = dict()

        for i, l in enumerate(self._ulabels):
            for k in range(self.conf.kfold_config.K):
                mask = np.logical_and(labels == l, self._pkl_no == k)
                l_idxs = np.nonzero(mask)
                if i in label_to_splits:
                    label_to_splits[i] = [None for _ in range(self.conf.kfold_config.K)]
                label_to_splits[i][k] = l_idxs

        return label_to_splits

    def split_unlabeled(self) -> List[np.ndarray]:
        labeled_mask = self.get_labeled_mask()
        split_unlabeled = list()
        for k in range(self.conf.kfold_config.K):
            mask = np.logical_and(~labeled_mask, self._pkl_no == k)
            split_unlabeled.append(np.nonzero(mask))

        return split_unlabeled