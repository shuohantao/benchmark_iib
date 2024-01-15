import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import ConfigManager
from utils.metrics import *
from utils.load import load_mnist
from modules.act_norm import ActNorm
import pickle
import torch.nn.functional as F
from tqdm import tqdm
cm = ConfigManager('config.yaml')
cm.config['model']['name'] = "res_sno"
res_sno = cm.get_model().to('cuda')
total_params = sum(p.numel() for p in res_sno.parameters())
print(f"Number of parameters: {total_params}")
res_sno.load_state_dict(torch.load('tmp/saved_ckpt/res_sno_epoch_0.pth'))
for i in res_sno.flows:
    if isinstance(i, ActNorm):
        i.is_initialized = True
train, test = load_mnist(batch_size=500, targets=list(range(10)), range=[56, 56])
val_samp = next(iter(test))[0]
res = [2*i for i in range(14, 28)]
versions = []
for resolution in res:
    downsampled = F.interpolate(val_samp, size=(resolution, resolution), mode='bilinear', align_corners=False)
    versions.append(downsampled)
versions.append(val_samp)
versions = [i.view(-1, np.prod(i.shape[-2:])).float() for i in versions]
mmds = []
for i, r in tqdm(enumerate(res)):
    X_samples = res_sno.sample(num_samples=500, resolution=r, device="cuda")
    X_samples = X_samples.view(-1, np.prod(X_samples.shape[-2:])).float().cpu()
    mmds.append(mmd(versions[i], X_samples).detach().cpu().numpy())
    del X_samples
with open(f'res_sno_mmds_{total_params}.pkl', 'wb') as f:
    pickle.dump(mmds, f)