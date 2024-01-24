import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import ConfigManager
from utils.metrics import *
from utils.load import load_mnist
from utils.load import load_model
from modules.act_norm import ActNorm
import pickle
import torch.nn.functional as F
from tqdm import tqdm

cm = ConfigManager('config.yaml')
cm.config['model']['name'] = cm.config['MMD_test']['test_model']
resolutions = cm.config['MMD_test']['resolutions']
model = cm.get_model().to('cuda')
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")
model = load_model(model, cm.config['MMD_test']['model_pth'])
train, test = load_mnist(batch_size=cm.config['MMD_test']['batch_size'], targets=list(range(10)), varying_shape=False, shape=resolutions[-1])
val_samp = next(iter(test))[0]
versions = []
for resolution in resolutions[:-1]:
    downsampled = F.interpolate(val_samp, size=(resolution, resolution), mode='bilinear', align_corners=False)
    versions.append(downsampled)
versions.append(val_samp)
versions = [i.view(-1, np.prod(i.shape[-2:])).float() for i in versions]
mmds = []
for i, r in tqdm(enumerate(resolutions)):
    X_samples = model.sample(num_samples=cm.config['MMD_test']['batch_size'], resolution=r, device="cuda")
    X_samples = X_samples.view(-1, np.prod(X_samples.shape[-2:])).float().cpu()
    mmds.append(mmd(versions[i], X_samples).detach().cpu().numpy())
mmds = [mmds, resolutions]
with open(f'{cm.config["MMD_test"]["model_pth"].split("/")[-1].split(".")[0]}_mmds_{total_params}.pkl', 'wb') as f:
    pickle.dump(mmds, f)