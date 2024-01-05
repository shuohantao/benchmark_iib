import matplotlib.pyplot as plt
import torch
import torchvision
def sample_grid(model, lowest, path):
    plt.rcParams['figure.figsize'] = [8, 8]
    size_range = range(lowest, lowest+4*4, 4)
    for i, size in enumerate(size_range):
        plt.subplot(2, 2, i+1)
        plt.title(f'{size}x{size}')
        _sample_image_grid_from_model(model, resolution=size, device=torch.device('cuda'))
    plt.savefig(path)

@torch.no_grad()
def _sample_image_grid_from_model(model, device, resolution=28, save_path=None):
    X_samples = model.sample(num_samples=4, resolution=resolution, device=device).cpu()
    image_shape = (1, resolution, resolution)
    data = X_samples.view(X_samples.shape[0], *(image_shape)) 
    grid = torchvision.utils.make_grid(data, nrow=2)
    plt.imshow(grid.permute(1, 2, 0))
    if save_path is not None:
        plt.savefig(save_path)