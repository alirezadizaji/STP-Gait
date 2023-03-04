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
    filterout_unlabeled: bool = True

class KFoldOperator(ABC, Generic[T]):
    r""" It initializes the train/test/validation sets, using KFold strategy.

    Args:
        K (int, optional): Number of KFolds in general. Defaults to `10`.
        init_valK (int, optional): Initial Kth number to assign validation. Defaults to `0`.
        init_testK (int, optional): Initial Kth number to assign test. Defaults to `1`.
        remove_labels (List[str], optional): Labels to be removed. Defaults to `list()`.
        filterout_unlabeled (bool, optional): If True, the ignore unlabeled samples for ever, O.W. consider them.
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.
    """
    def __init__(self, K:int = 10, init_valK: int = 0, init_testK: int = 1, 
            remove_labels: List[str] = list(), filterout_unlabeled: bool = True) -> None:
        self.K = K
        self._init_valK = init_valK
        self.valK = self._init_valK
        self.testK = init_testK

        self._filterout_unlabeled = filterout_unlabeled

        self.train: T = None
        self.val: T = None
        self.test: T = None

        labels = self.get_labels()
        labeled_mask = self.get_labeled_mask()
        self._ulabels = self.get_unique_labels()
        self._numeric_labels = self.get_numeric_labels()

        assert labels.size == labeled_mask.size, "Mismatch size between labels and the mask."

        num_samples = labels.size
        
        self._label_to_splits = self.split_labeled()
        self._unlabeled_to_splits = self.split_unlabeled()

        remained = (~self.get_ignore_mask()).sum()
        print(f"### {remained} out of {num_samples} remained after removing `{remove_labels}`. Remained labels: `{self._ulabels}` ###", flush=True)
        print(f"### {labeled_mask.sum()} out of {labels.size} are labeled. ###", flush=True)

    @abstractmethod
    def split_labeled() -> Dict[int, List[np.ndarray]]:
        """ It returns a list of indices per label, where each list is a separate fold. """

    @abstractmethod
    def split_unlabeled() -> List[np.ndarray]:
        """ It returns a list of indices for unlabeled samples, each one is usable as a separate fold. """

    @abstractmethod
    def get_ignore_mask() -> np.ndarray:
        """ It returns a mask which if it is False, that index must be ignored during data loading. """

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """ This returns label sets, to create Train/Val/Test KFolds """    

    @abstractmethod
    def get_numeric_labels(self) -> np.ndarray:
        """ Numeric conversion of labels to be used """

    @abstractmethod
    def get_labeled_mask(self) -> np.ndarray:
        """ This returns a mask, determining if a sample is labeled or not. """
    
    def get_unique_labels(self) -> np.ndarray:
        """ A list of unique labels used for the task. """
        
        labels = self.get_labels()
        labeled_mask = self.get_labeled_mask()
        ignore_mask = self.get_ignore_mask()
        
        mask1 = np.logical_and(~ignore_mask, labeled_mask)
        return np.unique(labels[mask1])

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

            # Assign labeled samples to separations
            for s in self._label_to_splits.values():
                val_indices.append(s[valK])
                test_indices.append(s[testK])

                for k in trainKs:
                    train_indices.append(s[k])
            
            if not self._filterout_unlabeled:
                # Assign unlabeled samples to separations
                u = self._unlabeled_to_splits
                for k in trainKs:
                    train_indices.append(u[k])
                val_indices.append(u[valK])
                test_indices.append(u[testK])

            val_indices = np.concatenate(val_indices).flatten()
            test_indices = np.concatenate(test_indices).flatten()
            train_indices = np.concatenate(train_indices).flatten()

            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)
            np.random.shuffle(train_indices)
            
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