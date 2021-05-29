from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class NA(Regularizer):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()


class TDURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(TDURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 2 + t ** 2)
            norm += torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        diff = torch.sqrt(ddiff**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)
