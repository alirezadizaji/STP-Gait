import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import proc_gait_data

class SkeletonDataset(Dataset):
    def __init__(self, 
            data: np.ndarray,
            names: np.ndarray,
            labels: np.ndarray) -> None:

        super().__init__()

        self.data = torch.from_numpy(data)
        self.names = names
        self.labels = torch.from_numpy(labels)

        assert self.data.size(0) == self.names.size == self.labels.size(0), f"Mismatch number of samples between data ({self.data.size()}), names ({names.size}) and labels ({self.labels.size(0)})."

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        return self.data[index], self.labels[index], self.names[index]

class DatasetInitializer:
    r""" It initializes the train/test/validation SkeletonDatasets, using KFold strategy.

    Args:
        load_dir (str): Where to load raw gait data; pickle format
        fillZ_empty (bool, optional): If `True`, then fill Z with zero value, O.W. process patient steps to get approximate Z locations. Defaults to `True`.
        filterout_unlabeled (bool, optional): If `True`, then filter out cases labeled as `unlabeled`. Defaults to `True`.
        K (int, optional): Number of KFolds in general. Defaults to `10`.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train/test/validation SkeletonDataset instances.
    """
    def __init__(self,
            load_dir: str,
            fillZ_empty: bool = True,
            filterout_unlabeled: bool = True,
            K: int = 10) -> None:

        self.K = K
        self.valK = 0
        self.testK = 1

        save_dir = os.path.join(load_dir, "processed.pkl")
        if not os.path.exists(save_dir):
            proc_gait_data(load_dir, save_dir, fillZ_empty)
        
        with open(save_dir, "rb") as f:
            import pickle
            data, labels, names = pickle.load(f)
        
        if filterout_unlabeled:
            mask = labels != 'unknown'
            data = data[mask]
            labels = labels[mask]
            names = names[mask]

        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._data = data[indices]
        self._labels = labels[indices]
        self._names = names[indices]

        # Split indices belong to each label into K subsets
        self._label_to_splits: Dict[int, List[np.ndarray]] = {}
        self._ulabels, self._label_indices = np.unique(self._labels, return_inverse=True)
        for i, _ in enumerate(self._ulabels):
            l_idxs = np.nonzero(self._label_indices == i)
            self._label_to_splits[i] = np.array_split(l_idxs[0], K)

    def __enter__(self):
        def _set_datasets():
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

            self.train = SkeletonDataset(self._data[train_indices], self._names[train_indices], 
                    self._label_indices[train_indices])
            self.val = SkeletonDataset(self._data[val_indices], self._names[val_indices], 
                    self._label_indices[val_indices])
            self.test = SkeletonDataset(self._data[test_indices], self._names[test_indices], 
                    self._label_indices[test_indices])

        _set_datasets()

        return None
     
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.train = self.test = self.val = None

        self.valK = (self.valK + 1) % self.K
        self.testK = (self.testK + 1) % self.K

        if self.valK == 0:
            print("@@@ WARNING: All KFolds have been iterated as validation and test sets evaluation.")


if __name__ == "__main__":
    initializer = DatasetInitializer("../../Data/")
    with initializer:
        print(initializer.test)
    print(initializer.test)