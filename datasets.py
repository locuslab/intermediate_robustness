import torch
from torchvision import datasets, transforms


dsets = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10
}


def get_train_loader(config):
    print(config.data.dataset)
    kwargs = {'num_workers': config.data.num_workers, 'pin_memory': True}
    train_transforms = []
    if config.data.training.flip_crop:
        train_transforms += [transforms.RandomCrop(size=32, padding=4), transforms.RandomHorizontalFlip()]
    train_transforms.append(transforms.ToTensor())
    train_transforms = transforms.Compose(train_transforms)
    train_loader = torch.utils.data.DataLoader(
        dsets[config.data.dataset](config.data.root, train=True, transform=train_transforms, download=True),
        batch_size=config.data.training.batch_size, shuffle=True, **kwargs)
    return train_loader


def get_test_loader(config):
    kwargs = {'num_workers': config.data.num_workers, 'pin_memory': True}
    test_transforms = [transforms.ToTensor()]
    test_transforms = transforms.Compose(test_transforms)
    test_loader = torch.utils.data.DataLoader(
        dsets[config.data.dataset](config.data.root, train=False, transform=test_transforms),
        batch_size=config.data.test.batch_size, shuffle=False, **kwargs)
    return test_loader
