from typing import Tuple
from tqdm import tqdm

from dig.xgraph.models import GCN_3l_BN
import numpy as np
from torch_geometric.data import Batch, DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..dataset.KFold.skeleton import KFoldSkeleton
from ..enums.separation import Separation

from ..models.transformer import SimpleTransformer
from .train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, np.ndarray]
OUT = Tuple[torch.Tensor, torch.Tensor]
C = float

class Entrypoint(TrainEntrypoint[IN, OUT, C]):
    def __init__(self) -> None:
        kfold = KFoldSkeleton(
            K=10,
            init_valK=0,
            init_testK=1,
            load_dir="../Data/output_1.pkl",
            fillZ_empty=True,
            filterout_unlabeled=True)
        model = SimpleTransformer(apply_loss_in_mask_loc=False)

        super().__init__(kfold, model)
    
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0]

        with torch.no_grad():
            N, T, _, _ = x.size()
            x = x.reshape(N, T, -1)                             # N, T, V, C -> N, T, L

            mask = torch.zeros(N, T).bool().to(x.device)
            if self.model.training:
                F = int(self.model._mask_ratio * T)
                idx1 = torch.arange(N).repeat(F)
                idx2 = torch.randint(0, T, size=(N, F)).flatten()
                mask[idx1, idx2] = True
            else:
                # For evaluation, mask only particular frames
                mask[:, 1::10] = True
            mask = mask.unsqueeze(2)

            # Just to make shapes OK
            mask = torch.logical_and(torch.ones_like(x).bool(), mask)
        
        x = self.model(x, mask)
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        loss = x[1]
        return loss

    def _train_start(self) -> None:
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

    def _train_epoch_end(self) -> None:
        print(f'epoch {self.epoch} loss value {np.mean(self.losses)}', flush=True)

    def _eval_epoch_end(self, datasep: Separation) -> C:
        print(f'epoch {self.epoch} separation {datasep} loss value {np.mean(self.losses)}', flush=True)
        return np.mean(self.losses)

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self.val_criterias[self.kfold.valK, self.epoch]
        return val <= self.val_criterias[self.kfold.valK, best_epoch]