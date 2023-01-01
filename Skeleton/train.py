from tqdm import tqdm

from dig.xgraph.models import GCN_3l_BN
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .dataset import DatasetInitializer

def _calc_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.log_softmax(x, dim=-1)
    loss = - torch.mean(x[torch.arange(x.size(0)), y])
    return loss

class Trainer:
    def __init__(self) -> None:
        self.dataset_initializer = DatasetInitializer("../../Data/")

    def train(self, train_loader: DataLoader) -> None:
        correct = total = 0
        total_loss = list()

        for i, data in tqdm(enumerate(train_loader)):
            # Ignore Z dimension
            data.x = data.x[..., [0, 1]] 
            x: torch.Tensor = self.model(data=data)
            loss = _calc_loss(x, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss.append(loss.item())
    
            y_pred = x.argmax(-1)
            correct += torch.sum(y_pred == data.y).item()
            total += data.y.numel()
        
            if i % 20 == 0:
                print(f'epoch {self.epoch_num} iter {i} loss value {np.mean(total_loss)}', flush=True)

        print(f'epoch {self.epoch_num} train acc {correct/total}', flush=True)

    def eval(self, loader, val: bool):
        name = "val" if val else "test"
        correct = total = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(loader)):
                # Ignore Z dimension
                data.x = data.x[..., [0, 1]]
                x: torch.Tensor = self.model(data=data)
                
                y_pred = x.argmax(-1)
                correct += torch.sum(y_pred == data.y).item()
                total += data.y.numel()
        
            print(f'epoch {self.epoch_num} {name} acc {correct/total}', flush=True)

    def run(self):
        self.model = GCN_3l_BN(model_level='graph', dim_node=2, dim_hidden=30, num_classes=12)
        self.optimizer = Adam(self.model.parameters(), 1e-2)
        epochs = 100
        
        with self.dataset_initializer:
            train_loader = DataLoader(self.dataset_initializer.train, batch_size=32, shuffle=True)
            val_loader = DataLoader(self.dataset_initializer.val, batch_size=32)
            test_loader = DataLoader(self.dataset_initializer.test, batch_size=32)

            for self.epoch_num in range(epochs):
                self.train(train_loader)
                self.eval(val_loader, True)
            
            self.eval(test_loader, False)
                