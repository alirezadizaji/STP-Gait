import os
from typing import List, Tuple
import pickle

import numpy as np
import torch

from ..dataset.KFold.skeleton import KFoldSkeleton
from ..enums.separation import Separation

from ..models.transformer import SimpleTransformer
from .train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]
OUT = Tuple[torch.Tensor, torch.Tensor]
C = float

class Entrypoint(TrainEntrypoint[IN, OUT, C]):
    def __init__(self) -> None:
        kfold = KFoldSkeleton(
            K=10,
            init_valK=0,
            init_testK=1,
            load_dir="../../Data/output_1.pkl",
            fillZ_empty=True,
            filterout_unlabeled=False)
        model = SimpleTransformer(apply_loss_in_mask_loc=False)
        model.to("cuda:0")
        super().__init__(kfold, model)
    
    def _model_forwarding(self, data: IN) -> OUT:
        x = data[0]
        x = x[..., [0, 1]]

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
                mask = data[2]
            mask = mask.unsqueeze(2)

            # Just to make shapes OK
            mask = torch.logical_and(torch.ones_like(x).bool(), mask)
        
        x = self.model(x.to("cuda:0"), mask.to("cuda:0"))
        return x

    def _calc_loss(self, x: OUT, data: IN) -> torch.Tensor:
        loss = x[1]
        return loss

    def _train_start(self) -> None:
        self.losses = list()

    def _eval_start(self) -> None:
        self._train_start()
        self.pred: List[np.ndarray] = list()
        self.names: List[np.ndarray] = list()

    def _train_iter_end(self, iter_num: int, loss: torch.Tensor, x: OUT, data: IN) -> None:
        self.losses.append(loss.item())

        if iter_num % 20 == 0:
            print(f'epoch {self.epoch} iter {iter_num} loss value {np.mean(self.losses)}', flush=True)

    def _eval_iter_end(self, iter_num: int, separation: Separation, loss: torch.Tensor, x: OUT, data: IN) -> None:
        if ~np.isnan(loss.item()):
            self.losses.append(loss.item())

        if separation == Separation.TEST:            
            self.names.append(self.test_loader.dataset.names[data[3]])
            self.pred.append(x[0].detach().cpu().numpy())
                
    def _train_epoch_end(self) -> None:
        print(f'epoch {self.epoch} loss value {np.mean(self.losses)}', flush=True)

    def _eval_epoch_end(self, datasep: Separation) -> C:
        print(f'epoch {self.epoch} separation {datasep} loss value {np.mean(self.losses)}', flush=True)

        # SAVE TEST outputs
        if datasep == Separation.TEST:
            save_dir = f"../Results/1_transformer/encoder_based/{self.kfold.testK}/output.pkl"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, 'wb') as f:
                pickle.dump((np.concatenate(self.pred), None, np.concatenate(self.names), None), f)

        return np.mean(self.losses)

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self.val_criterias[self.kfold.valK, self.epoch]
        return val <= self.val_criterias[self.kfold.valK, best_epoch]