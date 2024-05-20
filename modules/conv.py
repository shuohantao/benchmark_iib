#modified from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Advanced_Generative_Models/Normalizing_flows/advancednormflow.html
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F

class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        # use the Q matrix from QR decomposition as the initial weight to make sure it's invertible
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input, sample=False, **kwargs):
        _, _, height, width = input.shape

        logdet = (
            height * width * torch.log(torch.abs(torch.det(self.weight.squeeze())))
        )

        if not sample:
            out = F.conv2d(input, self.weight)
        else:
            out = F.conv2d(input, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            logdet = -logdet

        return out, logdet
    
class InvConv1d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        # use the Q matrix from QR decomposition as the initial weight to make sure it's invertible
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2)
        self.weight = nn.Parameter(weight)

    def forward(self, input, sample=False, **kwargs):
        S, B, C = input.shape

        logdet = (
            S * torch.log(torch.abs(torch.det(self.weight.squeeze())))
        )
        input = input.permute(1, 2, 0)
        if not sample:
            out = F.conv1d(input, self.weight)
        else:
            out = F.conv1d(input, self.weight.squeeze().inverse().unsqueeze(2))
            logdet = -logdet
        out = out.permute(2, 0, 1)
        return out, logdet