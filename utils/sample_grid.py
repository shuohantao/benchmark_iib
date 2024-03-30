import matplotlib.pyplot as plt
import torch
import torchvision
def sample_grid(model, lowest, resolutions, path, test_set, clip_range=None):
    model.eval()
    plt.rcParams['figure.figsize'] = [8, 8]
    if resolutions is None:
        size_range = range(lowest, lowest+4*4, 4)
    else:
        size_range = resolutions
    for i, size in enumerate(size_range):
        plt.subplot(2, 2, i+1)
        plt.title(f'{size}x{size}')
        _sample_image_grid_from_model(model, resolution=size, device=torch.device('cuda'), clip_range=clip_range, test_set=test_set)
    plt.savefig(path)
    model.train()
@torch.no_grad()
def _sample_image_grid_from_model(model, device, resolution=28, clip_range=None, test_set=None):
    if test_set is None:
        X_samples = model.sample(num_samples=4, resolution=resolution, device=device).cpu()
    else:
        X_samples = next(iter(test_set))[0][:4].to(device)
        X_samples = model.sample(X_samples).cpu()
    if clip_range is not None:
        X_samples = torch.clamp(X_samples, clip_range[0], clip_range[1])
    
    grid = torchvision.utils.make_grid(X_samples, nrow=2)
    plt.imshow(grid[0], cmap="gray", vmin=clip_range[0], vmax=clip_range[1])