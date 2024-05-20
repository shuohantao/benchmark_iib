import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys; sys.path.append(os.path.abspath(os.path.join('..'))); sys.path.append(os.path.abspath(os.path.join('.')))
from modules.RealNVP import RealNVP
from modules.resnet import ResNetMLP
class PatchVAE(nn.Module):
    def __init__(self, patch_size, n_hid):
        super(PatchVAE, self).__init__()
        self.mu_encoder = ResNetMLP(patch_size**2, 128, n_hid, 8)
        self.sig_encoder = ResNetMLP(patch_size**2, 128, n_hid, 8)
        self.sig_encoder.apply(self._init_weights)
        self.decoder = ResNetMLP(n_hid, 128, patch_size**2, 8)
        self.patch_size = patch_size
        self.n_hid = n_hid
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # Divide image into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size**2)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.patch_size**2)

        # Linear projection of patches
        patch_embeddings_mu = self.mu_encoder(patches)
        patch_embeddings_sig = F.relu(self.sig_encoder(patches))
        rec = self.decoder(patch_embeddings_mu+torch.randn_like(patch_embeddings_sig)*(0.5*patch_embeddings_sig).exp())
        rec = F.sigmoid(rec) * 255
        MSE = F.mse_loss(rec, patches, reduction='sum')
        KLD = -0.5 * torch.sum(1 + patch_embeddings_sig - patch_embeddings_mu.pow(2) - patch_embeddings_sig.exp())
        return MSE
    
    def revert_patches_to_image(self, patches, height, width):
        batch_size, num_patches, channels, patch_height, patch_width = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # [batch_size, channels, patch_height, patch_width, num_patches]
        patches = patches.contiguous().view(batch_size, channels, patch_height, patch_width, height // patch_height, width // patch_width)
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # [batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width]
        image = patches.contiguous().view(batch_size, channels, height, width)
        return image
    
    def sample(self, resolution, num_samples, device, **kwargs):
        patches = []
        for i in range(resolution**2//self.patch_size**2):
            base = torch.randn((num_samples, 1, self.n_hid)).to(device)
            patches.append((F.sigmoid(self.decoder(base))*255).view(num_samples, 1, self.patch_size, self.patch_size))
        patches = torch.stack(patches, 1)
        return self.revert_patches_to_image(patches, resolution, resolution)