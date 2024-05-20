import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

def sample_grid(model, lowest, resolutions, path, test_set=None, device='cuda', clip_range=None, num_context=4, sample_mean=False):
    model.eval()
    plt.rcParams['figure.figsize'] = [8, 8]
    if resolutions is None:
        size_range = range(lowest, lowest+4*4, 4)
    else:
        size_range = resolutions
    for i, size in enumerate(size_range):
        plt.subplot(2, 2, i+1)
        plt.title(f'{size}x{size}')
        _sample_image_grid_from_model(model, sample_mean = sample_mean, resolution=size, device=device, clip_range=clip_range, test_set=test_set, num_context=num_context)
    if path is not None:
        plt.savefig(path)
    model.train()
@torch.no_grad()
def _sample_image_grid_from_model(model, device, num_context, sample_mean, resolution=28, clip_range=None, test_set=None):
    if test_set is None:
        X_samples = model.sample(num_samples=4, resolution=resolution, device=device).cpu()
    else:
        batch = next(iter(test_set))[0]
        idx = torch.randint(0, batch.shape[0]-1, (4,))
        X_samples = batch[idx, ...].to(device)
        X_samples = F.upsample(X_samples, size=resolution, mode='bicubic', align_corners=False)
        X_samples = model.sample(X_samples, num_context, sample_mean, num_samples=4).cpu()
    # if clip_range is not None:
    #     X_samples = torch.clamp(X_samples, clip_range[0], clip_range[1])
    
    grid = torchvision.utils.make_grid(X_samples, nrow=2)
    if clip_range is not None:
        plt.imshow(grid[0], cmap="gray", vmin=clip_range[0], vmax=clip_range[1])
    else:
        plt.imshow(grid[0], cmap="gray")