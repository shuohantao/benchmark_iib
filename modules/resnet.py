import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x

        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip

        return x
    
class ResNet(nn.Module):
    """ResNet for scale and translate factors in Real NVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        out_channels (int): Number of channels in the output.
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
        padding (int): Padding for convolutional layers.
        double_after_norm (bool): Double input after input BatchNorm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_blocks, double_after_norm):
        super(ResNet, self).__init__()
        self.in_norm = nn.BatchNorm2d(in_channels)
        self.double_after_norm = double_after_norm
        self.in_conv = nn.Conv2d(2 * in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.in_skip = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)

        self.blocks = nn.ModuleList([ResidualBlock(mid_channels, mid_channels)
                                     for _ in range(num_blocks)])
        self.skips = nn.ModuleList([nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
                                    for _ in range(num_blocks)])

        self.out_norm = nn.BatchNorm2d(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x, **kwargs):
        x = self.in_norm(x)
        if self.double_after_norm:
            x *= 2.
        x = torch.cat((x, -x), dim=1)
        x = F.relu(x)
        x = self.in_conv(x)
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)
        x = self.out_norm(x_skip)
        x = F.relu(x)
        x = self.out_conv(x)

        return x
    
class ResBlockMLP(nn.Module):
    def __init__(self, in_shape, hid_shape):
        super(ResBlockMLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_shape, hid_shape),
                                nn.ReLU(),
                                nn.Linear(hid_shape, hid_shape),
                                nn.ReLU(),
                                nn.Linear(hid_shape, in_shape))
    def forward(self, x):
        return F.relu(x + self.fc(x))

class ResNetMLP(nn.Module):
    def __init__(self, in_shape, hid_shape, out_shape, n_blocks):
        super(ResNetMLP, self).__init__()
        self.fc = nn.Linear(in_shape, hid_shape)
        self.resblocks = nn.Sequential(*[ResBlockMLP(hid_shape, hid_shape) for _ in range(n_blocks)])
        self.out = nn.Linear(hid_shape, out_shape)
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.resblocks(x)
        return self.out(x)