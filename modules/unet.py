import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features)
        self.embed = nn.Linear(num_conditions, num_features * 2)

        # Initialize the embedding weights to zero
        self.embed.weight.data.fill_(0)
        self.embed.bias.data.fill_(0)

    def forward(self, x, condition):
        out = self.bn(x)
        gamma, beta = self.embed(condition).chunk(2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return out * (1 + gamma) + beta

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_size, downsample=True, keep=False):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if condition_size is not None:
            self.cbn1 = ConditionalBatchNorm2d(out_channels, condition_size)
            self.cbn2 = ConditionalBatchNorm2d(out_channels, condition_size)
        self.keep = keep
        if not keep:
            if downsample:
                self.downsampler = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.upsampler = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, condition=None):
        if condition is not None:
            x = F.relu(self.cbn1(self.conv1(x), condition))
            x = F.relu(self.cbn2(self.conv2(x), condition))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
        if not self.keep:
            if self.downsample:
                x = self.downsampler(x)
            else:
                x = self.upsampler(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_d, out_d, condition_size):
        super().__init__()
        self.encoder1 = UNetBlock(in_d, 64, condition_size, downsample=True)
        self.encoder2 = UNetBlock(64, 128, condition_size, downsample=True)
        self.bottleneck = UNetBlock(128, 256, condition_size, downsample=False, keep=True)
        self.decoder1 = UNetBlock(256, 128, condition_size, downsample=False)
        self.decoder2 = UNetBlock(128, 64, condition_size, downsample=False)
        self.final_conv = nn.Conv2d(64, out_d, kernel_size=3, padding=1)

    def forward(self, x, condition):
        # Condition should be a vector representing the noise scale
        x1 = self.encoder1(x, condition)
        x2 = self.encoder2(x1, condition)
        x3 = self.bottleneck(x2, condition)
        x4 = self.decoder1(x3, condition)
        x5 = self.decoder2(x4, condition)
        return self.final_conv(x5)

class UNetVanilla(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        condition_size = None
        self.encoder1 = UNetBlock(in_d, 64, condition_size, downsample=True)
        self.encoder2 = UNetBlock(64, 128, condition_size, downsample=True)
        self.bottleneck = UNetBlock(128, 256, condition_size, downsample=False, keep=True)
        self.decoder1 = UNetBlock(256, 128, condition_size, downsample=False)
        self.decoder2 = UNetBlock(128, 64, condition_size, downsample=False)
        self.final_conv = nn.Conv2d(64, out_d, kernel_size=3, padding=1)

    def forward(self, x):
        # Condition should be a vector representing the noise scale
        condition = None
        x1 = self.encoder1(x, condition)
        x2 = self.encoder2(x1, condition)
        x3 = self.bottleneck(x2, condition)
        x4 = self.decoder1(x3, condition)
        x5 = self.decoder2(x4, condition)
        return self.final_conv(x5)

# # Example usage
# if __name__ == "__main__":
#     noise_condition = torch.randn(1, 10)  # Example noise condition vector
#     input_image = torch.randn(1, 1, 256, 256)  # Example input image tensor
#     unet = UNet(1, 1, condition_size=10)
#     output_image = unet(input_image, noise_condition)
#     print(output_image.shape)  # Should print torch.Size([1, 3, 256, 256])
