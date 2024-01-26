import torch.nn as nn
import torch

class WaveletRes(nn.Module):
    def __init__(self, in_c, hid_c, out_c) -> None:
        super().__init__()
        self.c3 = nn.Conv2d(in_c, hid_c, 3, padding=1)
        self.c1 = nn.Conv2d(hid_c, out_c, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x_orig = x.clone()
        x = self.relu(self.c3(x))
        x = self.c1(x)
        return x + x_orig

class WaveFree(nn.Module):
    def __init__(self, in_c, hid_c, out_c) -> None:
        super().__init__()
        self.r1 = WaveletRes(hid_c, hid_c, hid_c)
        self.r2 = WaveletRes(hid_c, hid_c, hid_c)
        self.r3 = WaveletRes(hid_c, hid_c, hid_c)
        self.c1 = nn.Conv2d(in_c, hid_c, 1)
        self.c2 = nn.Conv2d(hid_c, out_c, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.c1(x)
        x = self.r1(x)
        x = self.relu(x)
        x = self.r2(x)
        x = self.relu(x)
        x = self.r3(x)
        x = self.relu(x)
        x = self.c2(x)
        return x