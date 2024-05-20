import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformers import *
from modules.coupling import CouplingPixelLayer
from modules.act_norm import ActNormSeq
from modules.conv import InvConv1d
from math import log2, exp

class PixelPatchTransformerFlow(nn.Module):
    def __init__(
        self, ndp, ndv, nde, nhead, num_flow_layers, num_encoder_layers, num_decoder_layers, mode, init_factor, num_rand_samples, context_num=4, patch_res=4):
        super(PixelPatchTransformerFlow, self).__init__()
        self.ndp = ndp
        self.ndv = ndv
        self.nde = nde
        self.num_layer = 32
        self.ndv_out = 1
        self.val_enc = ValueEncoder(ndv, nde)
        self.val_dec = ValueDecoder(2*nde, self.ndv_out)
        self.pos_enc = PositionEncoder(ndp, nde, mode, init_factor)
        Transformer = TransformerPrenorm
        self.transformer = Transformer(
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4*nde,
            d_model = 2*nde
        )
        
        self.context_num = context_num
        self.flows = nn.ModuleList([i for _ in range(num_flow_layers) for i in [ActNormSeq(2), InvConv1d(2), CouplingPixelLayer(ndv)]])
        self.prior = torch.distributions.Normal(0, 1)
        self.base_res = context_num
        self.query_token = nn.Parameter(torch.randn(1, 1, nde))
        self.patch_res = patch_res
        self.num_rand_samples = num_rand_samples

    def sample_flow(self, x):
        sig = x[..., :2].exp()
        mu = x[..., -2:]
        z = self.prior.sample(sig.shape).to(x.device)
        z = z * sig + mu
        for flow in reversed(self.flows):
            z, _ = flow(z, sample=True)
        z = F.sigmoid(z)
        return z

    def patchify(self, x):
        batch_size, channels, height, width = x.shape
        patch_size = height // self.patch_res
        # Divide image into patches
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_size**2)
        patches = patches.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, patch_size**2, channels).permute(1, 0, 2, 3)
        return patches
    
    def random_patchify(self, x, num_samples):
        batch_size, channels, height, width = x.shape
        patch_height = height // self.patch_res
        patch_width = width // self.patch_res
        patches = []
        
        # Calculate the possible positions for the top-left corner of patches
        max_y = height - patch_height//2
        max_x = width - patch_width//2
        centres = []
        for _ in range(num_samples):
            # Randomly select the top-left corner of the patch
            y_centre = torch.randint(patch_height//2, max_y + 1, (1,)).item()
            x_centre = torch.randint(patch_width//2, max_x + 1, (1,)).item()
            
            # Extract the patch
            patch = x[:, :, y_centre-patch_height//2:y_centre + patch_height//2, x_centre-patch_width//2:x_centre + patch_width//2]
            patches.append(patch.flatten(start_dim=2, end_dim=3).permute(0, 2, 1))
            centres.append((y_centre/height, x_centre/width))
        # Stack the patches into a tensor
        patches = torch.stack(patches, dim=0)  # Shape: (num_samples, batch_size, channels, patch_height, patch_width)
        
        return patches, centres
        
    def patchify_from_centre(self, x, centres):
        batch_size, channels, height, width = x.shape
        patch_height = height // self.patch_res
        patch_width = width // self.patch_res
        patches = []
        
        # Calculate the possible positions for the top-left corner of patches
        max_y = height - patch_height//2
        max_x = width - patch_width//2
        for c in centres:
            # Randomly select the top-left corner of the patch
            y_centre = int(c[0]*height)
            x_centre = int(c[1]*width)
            
            # Extract the patch
            patch = x[:, :, y_centre-patch_height//2:y_centre + patch_height//2, x_centre-patch_width//2:x_centre + patch_width//2]
            patches.append(patch.flatten(start_dim=2, end_dim=3).permute(0, 2, 1))
        # Stack the patches into a tensor
        patches = torch.stack(patches, dim=0)  # Shape: (num_samples, batch_size, channels, patch_height, patch_width)
        return patches
    
    def create_coord(self, x):
        batch_size, channel, height, width = x.shape
        x_coords = torch.arange(width, dtype=torch.float32, device=x.device).view(1, 1, 1, -1)
        y_coords = torch.arange(height, dtype=torch.float32, device=x.device).view(1, 1, -1, 1)
        pos = torch.zeros((batch_size, 2, height, width), dtype=torch.float32, device=x.device)
        pos[:, 0, :, :] = x_coords/width
        pos[:, 1, :, :] = y_coords/height
        return pos
    
    def revert_patches_to_image(self, patches, height, width):
        batch_size, num_patches, channels, patch_height, patch_width = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # [batch_size, channels, patch_height, patch_width, num_patches]
        patches = patches.contiguous().view(batch_size, channels, patch_height, patch_width, height // patch_height, width // patch_width)
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # [batch_size, channels, num_patches_h, patch_height, num_patches_w, patch_width]
        image = patches.contiguous().view(batch_size, channels, height, width)
        return image

    def format_context(self, context_patches, i):
        seq_length = context_patches.shape[0]
        seq_length = int(seq_length**0.5)
        row = i // seq_length
        col = i % seq_length
        context_patches_square = context_patches.view(seq_length, seq_length, *context_patches.shape[1:])
        context_list = []
        if row == 0:
            upper_context = []
            lower_context = [context_patches_square[row+1, col]]
        elif row == seq_length - 1:
            lower_context = []
            upper_context = [context_patches_square[row-1, col]]
        else:
            upper_context = [context_patches_square[row-1, col]]
            lower_context = [context_patches_square[row+1, col]]
        context_list.extend(upper_context)
        context_list.extend(lower_context)
        if col == 0:
            left_context = []
            right_context = [context_patches_square[row, col+1]]
        elif col == seq_length - 1:
            right_context = []
            left_context = [context_patches_square[row, col-1]]
        else:
            left_context = [context_patches_square[row, col-1]]
            right_context = [context_patches_square[row, col+1]]
        context_list.extend(left_context)
        context_list.extend(right_context)
        context_list.append(context_patches[i])
        return torch.cat(context_list, dim=1)
    def forward(self, x):
        '''
        Args:
            sample_vals: nS X B X ndv
            sample_posns: nS X B X ndp
            query_posns: nQ X B X ndp
        Returns:
            vals: prob distribution over (nQ X B X ndv)
        '''
        x_high = x.clone()
        b, c, height, width = x.shape
        patch_height_high = height // self.patch_res
        patch_width_high = width // self.patch_res
        x = F.interpolate(x, size=(self.base_res, self.base_res), mode='bilinear')
        patch_height_low = self.base_res // self.patch_res
        patch_width_low = self.base_res // self.patch_res
        pos = self.create_coord(x)
        query_pos = self.create_coord(x_high)
        #generate patches
        pos_patches = self.patchify(pos)
        x_patches = self.patchify(x)
        query_patches = self.patchify(query_pos)
        truth_patches = self.patchify(x_high)
        ################
        # pos_patches, centres = self.random_patchify(pos, self.num_rand_samples)
        # x_patches = self.patchify_from_centre(x, centres)
        # query_patches = self.patchify_from_centre(query_pos, centres)
        # truth_patches = self.patchify_from_centre(x_high, centres)
        ################
        x_patches = self.val_enc(x_patches)
        pos_patches = self.pos_enc(pos_patches)
        query_patches = self.pos_enc(query_patches)
        context_patches = torch.cat([x_patches, pos_patches], dim=-1)
        pred_patches = []
        for i in range(x_patches.shape[0]):
            #context_patch = self.format_context(context_patches, i).permute(1, 0, 2)
            context_patch = context_patches[i].permute(1, 0, 2)
            query_patch = query_patches[i].permute(1, 0, 2)
            query_patch = torch.cat([self.query_token.expand(query_patch.shape[0], query_patch.shape[1], query_patch.shape[2]), query_patch], dim=-1)
            pred_vals = self.transformer(context_patch, query_patch)
            pred_vals = self.val_dec(pred_vals)
            # print(pred_vals.shape)
            # pred_vals = self.sample_flow(pred_vals)
            pred_patches.append(pred_vals.permute(1, 0, 2))
        pred_patches = torch.stack(pred_patches, dim=0)
        return F.mse_loss(pred_patches, truth_patches)
    
    def sample(self, x, num_context, autoregessive=False, num_samples=-1, *args, **kwargs):
        x = x[:num_samples]
        x_high = x.clone()
        b, c, height, width = x.shape
        x = F.interpolate(x, size=(num_context, num_context), mode='bilinear')
        pos = self.create_coord(x)
        query_pos = self.create_coord(x_high)
        #generate patches
        pos_patches = self.patchify(pos)
        x_patches = self.patchify(x)
        query_patches = self.patchify(query_pos)
        truth_patches = self.patchify(x_high)
        ################
        # pos_patches, centres = self.random_patchify(pos, self.num_rand_samples)
        # x_patches = self.patchify_from_centre(x, centres)
        # query_patches = self.patchify_from_centre(query_pos, centres)
        # truth_patches = self.patchify_from_centre(x_high, centres)
        ################
        x_patches = self.val_enc(x_patches)
        pos_patches = self.pos_enc(pos_patches)
        query_patches = self.pos_enc(query_patches)
        context_patches = torch.cat([x_patches, pos_patches], dim=-1)
        pred_patches = []
        for i in range(x_patches.shape[0]):
            #context_patch = self.format_context(context_patches, i).permute(1, 0, 2)
            context_patch = context_patches[i].permute(1, 0, 2)
            query_patch = query_patches[i].permute(1, 0, 2)
            query_patch = torch.cat([self.query_token.expand(query_patch.shape[0], query_patch.shape[1], query_patch.shape[2]), query_patch], dim=-1)
            pred_vals = self.transformer(context_patch, query_patch)
            pred_vals = self.val_dec(pred_vals)
            # pred_vals = self.sample_flow(pred_vals)
            pred_patches.append(pred_vals.permute(1, 2, 0).view(b, c, height//self.patch_res, width//self.patch_res))
        pred_patches = torch.stack(pred_patches, dim=1)
        rec = self.revert_patches_to_image(pred_patches, height, width)
        return rec