import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.datasets import CIFAR10
import torch
import numpy as np
class _Collate_fn(object):
    def __init__(self, range):
        self.range = [i//4 for i in range]
    def __call__(self, batch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).float()),
            transforms.Resize(np.random.randint(*self.range)*4, antialias=True),
        ])
        data, labels = zip(*batch)
        data = [transform(item) for item in data]
        return torch.stack(data), torch.tensor(labels)
def load_mnist(batch_size=64, num_workers=0, targets=[0, 1], dir="data/mnist", varying_shape=False, range=None, **kwargs):
    if not varying_shape:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).float()),  # Quantize
            transforms.Resize(range[0], antialias=True),
        ])
        train_dataset = datasets.MNIST(
            root=dir,
            train=True,
            download=True,
            transform=transform,
        )
        indices = torch.zeros_like(train_dataset.targets, dtype=torch.bool)
        for t in targets:
            indices = indices | (train_dataset.targets == t)
        train_dataset.data, train_dataset.targets = train_dataset.data[indices], train_dataset.targets[indices]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
        test_dataset = datasets.MNIST(
            root=dir,
            train=False,
            download=True,
            transform=transform,
        )
        indices = torch.zeros_like(test_dataset.targets, dtype=torch.bool)
        for t in targets:
            indices = indices | (test_dataset.targets == t)
        test_dataset.data, test_dataset.targets = test_dataset.data[indices], test_dataset.targets[indices]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        train_dataset = datasets.MNIST(
            root=dir,
            train=True,
            download=True,
        )
        indices = torch.zeros_like(train_dataset.targets, dtype=torch.bool)
        for t in targets:
            indices = indices | (train_dataset.targets == t)
        train_dataset.data, train_dataset.targets = train_dataset.data[indices], train_dataset.targets[indices]
        assert range is not None, "Varying shape but failed to provide a range."
        collate_fn = _Collate_fn(range)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        test_dataset = datasets.MNIST(
            root=dir,
            train=False,
            download=True,
        )
        indices = torch.zeros_like(test_dataset.targets, dtype=torch.bool)
        for t in targets:
            indices = indices | (test_dataset.targets == t)
        test_dataset.data, test_dataset.targets = test_dataset.data[indices], test_dataset.targets[indices]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader
def load_cifar(batch_size=64, num_workers=0, targets=[0, 1], dir="data/cifar", varying_shape=False, range=None, **kwargs):
    trainset = CIFAR10(dir, train=True, download=True, transform=transform)
    testset = CIFAR10(dir, download=True, transform=transform)
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
    if not varying_shape:
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 255).float()),  # Quantize
                transforms.Resize(range[0], antialias=True),
            ])
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        assert range is not None, "Varying shape but failed to provide a range."
        collate_fn = _Collate_fn(range)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    return train_loader, test_loader
