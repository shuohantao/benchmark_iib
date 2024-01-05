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

    def forward(self, input, sample=False):
        _, _, height, width = input.shape

        # You can also use torch.slogdet(self.weight)[1] to summarize the operations below\n",
        logdet = (
            height * width * torch.log(torch.abs(torch.det(self.weight.squeeze())))
        )

        if not sample:
            out = F.conv2d(input, self.weight)
        else:
            out = F.conv2d(input, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))
            logdet = -logdet

        return out, logdet