from abc import abstractmethod, ABC

from torch import nn, Tensor

class Encoder(nn.Module, ABC):

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """ Output dimension of transformer encoder """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """ Forward function takes input feature """