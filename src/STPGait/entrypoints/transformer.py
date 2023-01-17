import os
from typing import List, Tuple
import pickle

import numpy as np
import torch

from ..config import BaseConfig, TrainingConfig
from ..dataset.KFold import SkeletonKFoldOperator, SkeletonKFoldConfig, KFoldConfig
from ..enums import Optim, Separation

from ..models.transformer import SimpleTransformer
from .train import TrainEntrypoint

IN = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]
OUT = Tuple[torch.Tensor, torch.Tensor]
C = float

class Entrypoint(TrainEntrypoint[IN, OUT, C, BaseConfig]):
    def __init__(self) -> None:
        kfold = SkeletonKFoldOperator(
            config=SkeletonKFoldConfig(
                kfold_config=KFoldConfig(K=10, init_valK=0, init_testK=1),
                load_dir="../../Data/output_1.pkl",
                filterout_unlabeled=False)
            )
        config = BaseConfig(
            try_num=1,
            try_name="transformer",
            device="cuda:0",
            eval_batch_size=1,
            save_log_in_file=True,
            training_config=TrainingConfig(num_epochs=200, optim_type=Optim.ADAM, lr=3e-3, early_stop=50)
        )
        super().__init__(kfold, config)
    
    def get_model(self):
        model = SimpleTransformer(apply_loss_in_mask_loc=False)
        return model
        
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
        
        x = self.model(x.to(self.conf.device), mask.to(self.conf.device))
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
            names = self.test_loader.dataset.names[data[3]]
            if isinstance(names, str):
                names = [names]
            else:
                names = names.tolist()
            self.names = self.names + names
            self.pred.append(x[0].detach().cpu().numpy())
                
    def _train_epoch_end(self) -> None:
        print(f'epoch {self.epoch} loss value {np.mean(self.losses)}', flush=True)

    def _eval_epoch_end(self, datasep: Separation) -> C:
        print(f'epoch {self.epoch} separation {datasep} loss value {np.mean(self.losses)}', flush=True)

        # SAVE TEST outputs
        if datasep == Separation.TEST:
            save_dir = os.path.join(self.conf.save_dir, "encoder_based", str(self.kfold.testK), "output.pkl")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            with open(save_dir, 'wb') as f:
                pred = np.concatenate(self.pred)
                # N, T, V*C -> N, T, V, C
                pred = np.stack(np.array_split(pred, self.test_loader.dataset.V, axis=2), axis=2)
                pickle.dump((np.concatenate(pred), None, np.array(self.names), None), f)

        return np.mean(self.losses)

    def best_epoch_criteria(self, best_epoch: int) -> bool:
        val = self.val_criterias[self.kfold.valK, self.epoch]
        return val <= self.val_criterias[self.kfold.valK, best_epoch]