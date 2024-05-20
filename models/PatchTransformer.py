import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys; sys.path.append(os.path.abspath(os.path.join('..'))); sys.path.append(os.path.abspath(os.path.join('.')))
from modules.RealNVP import RealNVP
from modules.resnet import ResNetMLP
class PatchTransformer(nn.Module):
    def __init__(self, patch_size, dim, depth, heads, channels=1, beta=0.0):
        super(PatchTransformer, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.patch_dim = channels * patch_size * patch_size

        self.linear_projection = nn.Linear(self.patch_dim, dim)
        self.position_embedding = nn.Linear(2, dim)  # Linear layer for positional encoding

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=2*dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_projection = nn.Linear(dim, 2*self.patch_dim)  # Project back to patch dimension
        # self.flow = RealNVP(self.patch_size)
        self.vae_decoder = ResNetMLP(self.patch_dim, 64, self.patch_dim, 6)
        self.prior = torch.distributions.Normal(0, 1)
        self.beta = beta
        self.start_token = nn.Parameter(torch.randn(1, 1, dim))
        self.query_token = nn.Parameter(torch.randn(1, 1, dim))
    def generate_patch_coords(self, height, width):
        """ Generate normalized coordinates for the center of each patch. """
        patches_across_h = height // self.patch_size
        patches_across_w = width // self.patch_size
        y_coords = torch.linspace(self.patch_size/(2*height), 1-self.patch_size/(2*height), patches_across_h)
        x_coords = torch.linspace(self.patch_size/(2*width), 1-self.patch_size/(2*width), patches_across_w)
        y, x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack((x, y), -1).view(-1, 2)  # Flatten and combine
        return coords
    
    def revert_patches_to_image(self, patches, height, width):
        batch_size, num_patches, channels, patch_height, patch_width = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # [batch_size, channels, patch_height, patch_width, num_patches]
        patches = patches.contiguous().view(batch_size, channels, patch_height, patch_width, height // patch_height, width // patch_width)
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # [batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width]
        image = patches.contiguous().view(batch_size, channels, height, width)
        return image

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Divide image into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size**2)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.patch_dim)

        # Linear projection of patches
        patch_embeddings = self.linear_projection(patches)

        # Get positional embeddings from normalized coordinates
        coords = self.generate_patch_coords(height, width).to(x.device)
        coords_embeddings = self.position_embedding(coords.float())
        coords_embeddings = coords_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Broadcast to batch size

        # Add positional embeddings to patch embeddings
        patch_embeddings += coords_embeddings

        perm = torch.randperm(patch_embeddings.shape[1])

        patch_embeddings = patch_embeddings.permute(1, 0, 2)

        patch_embeddings = patch_embeddings[perm]

        coords_embeddings = coords_embeddings.permute(1, 0, 2)

        coords_embeddings = coords_embeddings[perm]

        start_token = self.start_token.expand(-1, batch_size, -1)

        patch_embeddings = torch.cat((start_token, patch_embeddings), dim=0)

        # Permute for transformer input
        # Transformer expects [seq_len, batch_size, dim]

        patches = patches.permute(1, 0, 2)[perm]

        sll = 0
        for i in range(patch_embeddings.shape[0]-1):
            context = patch_embeddings[:i+1]
            query = coords_embeddings[i].unsqueeze(0)
            query = self.query_token + query
            input = torch.cat((context, query), dim=0)
            condition = self.output_projection(self.transformer(input)[-1])
            condition = condition.view(batch_size, 2, -1)
            mu, sig = torch.chunk(condition, 2, dim=1)
            sample = self.prior.sample(mu.shape).to(mu.device)*(0.5*sig).exp() + mu
            sample = F.sigmoid(self.vae_decoder(sample))*255
            ground_truth = patches[i].view(batch_size, channels, -1)
            mse_loss = F.mse_loss(sample, ground_truth)
            kl_loss = -0.5 * torch.sum(1 + sig - sig.exp())
            sll += mse_loss + self.beta*kl_loss

        return sll
    
    def sample(self, resolution, num_samples, device, **kwargs):
        coords = self.generate_patch_coords(resolution, resolution).to(device)
        batch_size = num_samples
        coords_embeddings = self.position_embedding(coords.float())
        coords_embeddings = coords_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        coords_embeddings = coords_embeddings.permute(1, 0, 2)

        start_token = self.start_token.expand(batch_size, -1, -1)

        context = start_token.permute(1, 0, 2)
        img = []
        for i in range((resolution//self.patch_size)**2):
            query = coords_embeddings[i].unsqueeze(0)
            query = self.query_token + query
            input = torch.cat((context, query), dim=0)
            condition = self.output_projection(self.transformer(input)[-1]).view(batch_size, 2, -1)
            mu, sig = torch.chunk(condition, 2, dim=1)
            sample = self.prior.sample(mu.shape).to(mu.device)*sig.exp() + mu
            sample = F.sigmoid(self.vae_decoder(sample))*255
            img.append(sample.view(batch_size, 1, self.patch_size, self.patch_size))
            sample = sample.permute(1, 0, 2)
            sample = self.linear_projection(sample)
            context = torch.cat((context, sample), dim=0)
        img = torch.stack(img, dim=1)
        img = self.revert_patches_to_image(img, resolution, resolution)
        return img