import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Dequantization(nn.Module):
    def __init__(self, quants = 256, epsilon = 1e-5):
        super().__init__()
        self.quants = quants
        self.epsilon = epsilon
    def forward(self, x, sample = False):
        if not sample:
            x, ldj_1 = self.dequant(x)
            x, ldj_2 = self.sigmoid(x, sample=sample)
            ldj = ldj_1 + ldj_2
        else:
            x, ldj = self.sigmoid(x, sample=sample)
            x = x * self.quants
            ldj += np.prod(x.shape[1:]) * np.log(self.quants)
            x = x.floor().clamp(0, self.quants - 1).int()
        return x, ldj
    def dequant(self, x):
        x = x.float()
        x = x + torch.rand_like(x)
        x = x / self.quants
        ldj = - np.prod(x.shape[1:]) * np.log(self.quants)
        return x, ldj
    def sigmoid(self, x, sample=False):
        if not sample:
            x = x * (1 - self.epsilon) + 0.5 * self.epsilon
            ldj = np.prod(x.shape[1:]) * np.log(1 - self.epsilon)
            ldj += -torch.log(x*(1-x)).sum(dim=[i + 1 for i in range(x.dim() - 1)])
            x = torch.log(x) - torch.log(1-x)
        else:
            ldj = (-x-2*F.softplus(-x)).sum(dim=[1,2,3])
            x = torch.sigmoid(x)
            ldj -= np.prod(x.shape[1:]) * np.log(1 - self.epsilon)
            x = (x - 0.5 * self.epsilon) / (1 - self.epsilon)
        return x, ldj