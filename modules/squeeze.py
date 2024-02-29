import torch.nn as nn
import torch
class SqueezeFlow(nn.Module):
    def forward(self, z, sample=False, **kwargs):
        B, C, H, W = z.shape
        if not sample:
            # Forward direction: H x W x C => H/2 x W/2 x 4C
            z = z.reshape(B, C, H//2, 2, W//2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4*C, H//2, W//2)
        else:
            # Reverse direction: H/2 x W/2 x 4C => H x W x C
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        return z, torch.zeros([]).to('cuda')
class PatchFlow(nn.Module):
    def __init__(self, p_num):
        super().__init__()
        self.p_num = p_num
    def forward(self, z, sample=False):
        B, C, H, W = z.shape
        if not sample:
            patch_size = H//self.p_num
            patches = z.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, patch_size, patch_size)
            patches = patches.reshape(B, -1, patch_size, patch_size)
        else:
            p_num = self.p_num
            patch_size = H
            C = C//self.p_num**2
            patches = z.reshape(B, p_num, p_num, C, patch_size, patch_size)
            patches = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H*p_num, W*p_num)
        return patches, 0