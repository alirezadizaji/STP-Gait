from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, 
            X: np.ndarray,
            names: np.ndarray,
            Y: np.ndarray,
            labeled: np.ndarray,
            frame_invalid: Optional[np.ndarray] = None) -> None:
        """ Init function of Skeleton Dataset

        Args:
            X (np.ndarray): Features of each sequence (shape: N, T, V, C).
            names (np.ndarray): Names of each sequence (shape: N).
            Y (np.ndarray): Labels of each sequence (shape: N).
            labeled (np.ndarray): A boolean mask; If True, then labeled (shape: N).
            frame_invalid (Optional[np.ndarray], optional): If an element is True, then that frame is invalid, O.W. valid (shape: N, T). Defaults to None.
        """
        super().__init__()
        self.X: torch.Tensor = torch.from_numpy(X).float()      # N, T, V, C
        self.Y: torch.Tensor = torch.from_numpy(Y)              # N
        self.labeled: torch.Tensor = torch.from_numpy(labeled)  # N

        if frame_invalid is None:
            frame_invalid = np.zeros((self.X.size(0), self.X.size(1)), dtype=np.bool)
        self.frame_invalid: torch.Tensor = torch.from_numpy(frame_invalid)        # N, T
        
        self.names = names                                      # N

        assert self.X.size(0) == self.names.size == self.Y.size(0) == self.frame_invalid.size(0), f"Mismatch number of samples between data ({X.size(0)}), names ({names.size}), labels ({Y.size(0)}), and frame invalidity ({self.frame_invalid.size(0)})."
        
        self.V = self.X.size(2)
        
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        return self.X[index], self.Y[index], self.frame_invalid[index], index, self.labeled[index]