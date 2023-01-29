from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np
from torch.utils.data import Dataset

from ...enums import Separation

T = TypeVar('T', bound=Dataset)

@dataclass
class KFoldConfig:
    K: int = 10
    init_valK: int = 0
    init_testK: int = 1
    remove_labels: List[str] = field(default_factory=list)

IGNORE = "IgNoRe"

class KFoldOperator(ABC, Generic[T]):
    r""" It initializes the train/test/validation sets, using KFold strategy.

    Args:
        K (int, optional): Number of KFolds in general. Defaults to `10`.
        init_valK (int, optional): Initial Kth number to assign validation. Defaults to `0`.
        init_testK (int, optional): Initial Kth number to assign test. Defaults to `1`.
        remove_labels (List[str], optional): Labels to be removed. Defaults to `list()`.
        
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.
    """
    def __init__(self, K:int = 10, init_valK: int = 0, init_testK: int = 1, remove_labels: List[str] = list()) -> None:
        self.K = K
        self._init_valK = init_valK
        self.valK = self._init_valK
        self.testK = init_testK

        self.train: T = None
        self.val: T = None
        self.test: T = None

        # Split indices belong to each label into K subsets
        self._label_to_splits: Dict[int, List[np.ndarray]] = {}
        labels = self.get_labels()
        num_samples = labels.size
        
        ignore_mask = np.zeros_like(labels, dtype=np.bool)
        if remove_labels:
            ignore_mask = labels[:, np.newaxis] == np.array(remove_labels)[np.newaxis, :]
            ignore_mask = ignore_mask.sum(1) > 0

        self._ulabels = np.unique(labels[~ignore_mask])
        self._numeric_labels = np.full_like(labels, fill_value=-np.inf)

        for i, l in enumerate(self._ulabels):
            l_idxs = np.nonzero(labels == l)
            self._numeric_labels[l_idxs] = i
            self._label_to_splits[i] = np.array_split(l_idxs[0], K)
        
        remained = (~ignore_mask).sum()
        print(f"### {remained} out of {num_samples} remained after removing `{remove_labels}`. Remained labels: `{self._ulabels}` ###", flush=True)

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """ This returns label sets, to create Train/Val/Test KFolds """    

    @abstractmethod
    def set_sets(self, train_indices: np.ndarray, val_indices: np.ndarray, 
            test_indices: np.ndarray) -> Dict[Separation, T]:
        """ It sets train, validation and test sets, using the separated indices

        Args:
            train_indices (np.ndarray): train set indices
            val_indices (np.ndarray): validation set indices
            test_indices (np.ndarray): test set indices

        Returns:
            (Dict[Separation, T]): Dataset instance per separation
        """

    def __enter__(self):
        def _get_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """ It returns train, validation and test sets indices to split with """

            Ks = np.arange(self.K)
            valK = (self.valK + self.K) % self.K
            testK = (self.testK + self.K) % self.K
            trainKs = np.delete(Ks, [valK, testK])

            val_indices = []
            train_indices = []
            test_indices = []

            for s in self._label_to_splits.values():
                val_indices.append(s[valK])
                test_indices.append(s[testK])

                for k in trainKs:
                    train_indices.append(s[k])
            
            val_indices = np.concatenate(val_indices).flatten()
            test_indices = np.concatenate(test_indices).flatten()
            train_indices = np.concatenate(train_indices).flatten()

            return train_indices, val_indices, test_indices

        train_indices, val_indices, test_indices = _get_indices()
        out = self.set_sets(train_indices, val_indices, test_indices)
        
        self.train = out[Separation.TRAIN]
        self.val = out[Separation.VAL]
        self.test = out[Separation.TEST]

        print(f"@@@ K(={self.K})Fold applied for SkeletonDataset: valK: {self.valK}, testK: {self.testK}. @@@", flush=True)
        
        return None
     
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.train = self.test = self.val = None

        self.valK = (self.valK + 1) % self.K
        self.testK = (self.testK + 1) % self.K

        if self.valK == self._init_valK:
            print("@@@ WARNING: All KFolds have been iterated as validation and test sets evaluation. @@@", flush=True)