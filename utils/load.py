import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.datasets import CIFAR10
import torch
import numpy as np
from models.CAF import CAF
from models.CAFMod import CAFMod
from models.CAFFNO import CAFFNO
from models.ResSNO import ResSNO
from models.WaveletFlow import WaveletFlow
from modules.act_norm import ActNorm
import numpy as np
from data.darcy.darcy_dataset import DarcyDataset

class _Collate_fn(object):
    def __init__(self, shape_setting, upper, lower):
        self.percentages = [i[1] for i in shape_setting]
        self.res = [i[0] for i in shape_setting]
        self.upper = upper
        self.lower = lower
    def __call__(self, batch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * (self.upper-self.lower) - self.lower).float()),
            transforms.Resize(int(np.random.choice(self.res, p=self.percentages)), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 255)),
        ])
        data, labels = zip(*batch)
        data = [transform(item) for item in data]
        return torch.stack(data), torch.tensor(labels)
class _Collate_fn_darcy(object):
    def __init__(self, shape_setting):
        self.percentages = [i[1] for i in shape_setting]
        self.res = [i[0] for i in shape_setting]

    def __call__(self, batch):
        transform = transforms.Compose([
            transforms.Resize(int(np.random.choice(self.res, p=self.percentages)), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC),
        ])
        batch = [transform(item) for item in batch]
        return torch.stack(batch), None
    
def load_mnist(batch_size=64, num_workers=0, targets=[0, 1], path="data/mnist", shape_setting=None, upper=255, lower=0, **kwargs):
    train_dataset = datasets.MNIST(
        root=path,
        train=True,
        download=True,
    )
    indices = torch.zeros_like(train_dataset.targets, dtype=torch.bool)
    for t in targets:
        indices = indices | (train_dataset.targets == t)
    train_dataset.data, train_dataset.targets = train_dataset.data[indices], train_dataset.targets[indices]
    assert shape_setting is not None, "Varying shape but failed to provide shapes."
    collate_fn = _Collate_fn(shape_setting, upper, lower)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    test_dataset = datasets.MNIST(
        root=path,
        train=False,
        download=True,
    )
    indices = torch.zeros_like(test_dataset.targets, dtype=torch.bool)
    for t in targets:
        indices = indices | (test_dataset.targets == t)
    test_dataset.data, test_dataset.targets = test_dataset.data[indices], test_dataset.targets[indices]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader
def load_cifar(batch_size=64, num_workers=0, targets=[0, 1], dir="data/cifar", range=None, **kwargs):
    trainset = CIFAR10(dir, train=True, download=True)
    testset = CIFAR10(dir, download=True)
    idx = []
    idx_test = []
    targets = set(targets)
    for i in range(len(trainset)):
        current_class = trainset[i][1]
        if current_class in targets:
            idx.append(i)
    for i in range(len(testset)):
        current_class = testset[i][1]
        if current_class in targets:
            idx_test.append(i)
    trainset = Subset(trainset, idx)
    testset = Subset(testset, idx_test)
    assert range is not None, "Varying shape but failed to provide a range."
    collate_fn = _Collate_fn(range)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader

def load_model(model, path):
    
    model.load_state_dict(torch.load(path), strict=False)
    # if isinstance(model, CAF) or isinstance(model, CAFMod) or isinstance(model, CAFFNO):
        # for i in model.flow:
        #     if isinstance(i, ActNorm):
        #         i.is_initialized = True
        # for i in model.ar_flow:
        #     if isinstance(i, ActNorm):
        #         i.is_initialized = True
    if isinstance(model, ResSNO):
        for i in model.flows:
            if isinstance(i, ActNorm):
                i.is_initialized = True
    if isinstance(model, WaveletFlow):
        for i in model.flows:
            for j in i:
                if isinstance(j, ActNorm):
                    j.is_initialized = True
        for i in model.uncon_flow:
            if isinstance(i, ActNorm):
                i.is_initialized = True
    return model

def load_darcy(batch_size=64, num_workers=0, path="data/darcy", num_samples=-1, shape_setting=None, **kwargs):
    train_dataset = DarcyDataset(path, num_samples=num_samples)
    assert shape_setting is not None, "Varying shape but failed to provide shapes."
    collate_fn = _Collate_fn_darcy(shape_setting)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    test_dataset = train_dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader