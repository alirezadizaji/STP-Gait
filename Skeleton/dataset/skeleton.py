import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from ..context.skeleton import Skeleton
from ..data import proc_gait_data

class SkeletonDataset(Dataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray) -> None:

        super().__init__()

        lower_part_node_indices = np.delete(np.arange(Skeleton.num_nodes), Skeleton.upper_part_node_indices)
        X = X[..., lower_part_node_indices, :]

        X: torch.Tensor = torch.from_numpy(X).float()
        Y: torch.Tensor = torch.from_numpy(Y)

        assert X.size(0) == names.size == Y.size(0), f"Mismatch number of samples between data ({X.size(0)}), names ({names.size}) and labels ({Y.size(0)})."

        N, T, V, C = X.size()
        X = X.reshape(N, T * V, C)
        edge_index = Skeleton.get_simple_interframe_edges(T, use_lower_part=True)
        self.data: List[Data] = [Data(x=x, edge_index=edge_index, y=y, name=n) 
            for x, n, y in zip(X, names, Y)]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> Batch:
        if not isinstance(index, list):
            index = [index]
        return [self.data[i] for i in index]

class DatasetInitializer:
    r""" It initializes the train/test/validation SkeletonDatasets, using KFold strategy.

    Args:
        load_dir (str): Where to load raw gait data; pickle format
        fillZ_empty (bool, optional): If `True`, then fill Z with zero value, O.W. process patient steps to get approximate Z locations. Defaults to `True`.
        filterout_unlabeled (bool, optional): If `True`, then filter out cases labeled as `unlabeled`. Defaults to `True`.
        K (int, optional): Number of KFolds in general. Defaults to `10`.
    
    NOTE:
        Each validation and test sets will get 1/K partial of the whole dataset.

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

        root_dir = os.path.dirname(load_dir)
        save_dir = os.path.join(root_dir, "processed.pkl")
        if not os.path.exists(save_dir):
            proc_gait_data(load_dir, root_dir, fillZ_empty)
        
        with open(save_dir, "rb") as f:
            import pickle
            x, labels, names = pickle.load(f)
        
        if filterout_unlabeled:
            mask = labels != 'unlabeled'
            x = x[mask]
            labels = labels[mask]
            names = names[mask]
        
        # Shuffle dataset
        indices = np.arange(labels.size)
        np.random.shuffle(indices)
        self._x = x[indices]
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

            self.train = SkeletonDataset(self._x[train_indices], self._names[train_indices], 
                    self._label_indices[train_indices])
            self.val = SkeletonDataset(self._x[val_indices], self._names[val_indices], 
                    self._label_indices[val_indices])
            self.test = SkeletonDataset(self._x[test_indices], self._names[test_indices], 
                    self._label_indices[test_indices])

        _set_datasets()
        print(f"@@@ K(={self.K})Fold applied for SkeletonDataset: valK: {self.valK}, testK: {self.testK}. @@@", flush=True)
        
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