import argparse
import json
import logging
import math
import itertools
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import *
from models.mnist import mnist_classifier
from models.preactresnet import preactresnet18
from datasets import get_train_loader, get_test_loader

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def reflect(x, a, b):
    if not torch.is_tensor(a):
        a = torch.zeros_like(x).fill_(a)
    if not torch.is_tensor(b):
        b = torch.zeros_like(x).fill_(b)
    while len(torch.where((x < a) | (x > b))[0]) > 0:
        low = torch.where(x < a)
        high = torch.where(x > b)
        x[low] = a[low] + (a[low] - x[low])
        x[high] = b[high] - (x[high] - b[high])
    return x


def epoch_standard(loader, model, opt=None, eval_mode=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for batch in loader:
        if type(batch) is dict:
            X, y = batch['input'], batch['target']
        else:
            X, y = batch
        X,y = X.cuda(), y.cuda()
        if X.shape[1] == 1:
            yp = model(X)
        else:
            yp = model(normalize(X))
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_discrete_random_sampling(loader, model, opt=None, p=1, m=20):
    padded = 40
    unpadded = 32
    total_loss = 0.
    for batch in loader:
        X, y = batch['input'], batch['target']
        X, y = X.cuda(), y.cuda()

        losses = torch.zeros(m, X.size(0)).cuda()
        max_loss = 0
        for i in range(m):
            scale_min, scale_max = 0.9, 1.1
            rotate_min, rotate_max = -10, 10

            flip = np.random.choice([True, False])
            scale = np.random.uniform(scale_min, scale_max)
            rotate = np.random.uniform(rotate_min, rotate_max)
            crop_range = math.floor(scale * padded) - unpadded
            crop_x = np.random.randint(0, crop_range + 1)
            crop_y = np.random.randint(0, crop_range + 1)

            sample = X.detach().clone()
            if flip: sample = TF.hflip(sample)
            sample = TF.resize(sample, size=(int(padded * scale), int(padded * scale)))
            sample = TF.rotate(sample, angle=rotate)
            sample = TF.crop(sample, top=crop_y, left=crop_x, height=unpadded, width=unpadded)
            yp = model(normalize(sample))
            loss = nn.CrossEntropyLoss(reduction='none')(yp, y)
            losses[i] = loss.detach()
        
        loss = losses.transpose(0, 1)
        loss = (torch.exp(torch.logsumexp(torch.log(loss + 1e-10) * p, dim=1)/ p) * (1/m)**(1/p))
        loss = loss.mean()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)    


def epoch_mcmc(loader, model, opt=None, p=1, m=20):
    """MCMC with discrete data augmentation transformations"""
    padded = 40
    unpadded = 32
    total_loss = 0.
    total_adv_loss = 0.
    
    for batch in loader:
        if type(batch) is dict:
            X, y = batch['input'], batch['target']
        X, y = X.cuda(), y.cuda()
    
        losses = torch.zeros(X.size(0), m)

        transforms = torch.zeros(X.shape[0], 5)
        scale_min, scale_max = 0.9, 1.1
        rotate_min, rotate_max = -10, 10

        flip = np.random.choice([True, False])
        scale = np.random.uniform(scale_min, scale_max)
        rotate = np.random.uniform(rotate_min, rotate_max)
        crop_range = math.floor(scale * padded) - unpadded
        crop_x = np.random.randint(0, crop_range + 1)
        crop_y = np.random.randint(0, crop_range + 1)

        previous = X.detach().clone()
        if flip: previous = TF.hflip(previous)
        previous = TF.resize(previous, size=(int(padded * scale), int(padded * scale)))
        previous = TF.rotate(previous, angle=rotate)
        previous = TF.crop(previous, top=crop_y, left=crop_x, height=unpadded , width=unpadded)

        transforms[:, 0] = int(flip)
        transforms[:, 1] = rotate
        transforms[:, 2] = crop_x
        transforms[:, 3] = crop_y
        transforms[:, 4] = scale

        thetas = np.linspace(0, p, m)
        max_loss = 0
        idxx = 0
        for i in range(len(thetas)):
            theta = thetas[i]
            for j in range(20):
                yp = model(normalize(previous))
                loss = nn.CrossEntropyLoss(reduction='none')(yp,y)
                if j == 19:
                    losses[:, i] = loss.detach().cpu()
                log_loss = theta * torch.log(loss + 1e-10)

                # transforms from previous iteration
                flip = transforms[:, 0]
                rotate = transforms[:, 1]
                crop_x = transforms[:, 2]
                crop_y = transforms[:, 3]
                scale = transforms[:, 4]

                # proposed deltas
                flip_delta = np.random.choice([True, False])
                rotate_delta = np.random.normal(0, 5)
                crop_x_delta = int(np.random.normal(0, 2))
                crop_y_delta = int(np.random.normal(0, 2))
                scale_delta = np.random.normal(0, 0.5)

                # proposed transforms with reflections
                if flip_delta:
                    flip_proposal = 1-flip
                else:
                    flip_proposal = flip
                scale_proposal = reflect((scale + scale_delta).float(), scale_min, scale_max)
                rotate_proposal = reflect((rotate + rotate_delta).float(), rotate_min, rotate_max)
                crop_range_max = (np.floor(scale_proposal * padded) - unpadded).int()
                crop_x_proposal = reflect((crop_x + crop_x_delta).int(), 0, crop_range_max)
                crop_y_proposal = reflect((crop_y + crop_y_delta).int(), 0, crop_range_max)

                new_transforms = torch.stack((flip_proposal.int(), rotate_proposal, crop_x_proposal, crop_y_proposal, scale_proposal), dim=1)
                proposal = X.detach().clone()
                transformed_proposal = torch.zeros(X.shape[0], X.shape[1], unpadded, unpadded)
                proposal[flip_proposal.bool()] = TF.hflip(proposal[flip_proposal.bool()])
                for idx in range(X.size(0)):
                    scaled = TF.resize(proposal[idx], size=(int(padded * scale_proposal[idx]), int(padded * scale_proposal[idx])))
                    rotated = TF.rotate(scaled, angle=rotate_proposal[idx].item())
                    cropped = TF.crop(rotated, top=crop_y_proposal[idx], left=crop_x_proposal[idx], height=unpadded, width=unpadded)
                    transformed_proposal[idx] = cropped

                proposal = transformed_proposal.cuda()
                yp_next = model(normalize(proposal))
                loss_next = nn.CrossEntropyLoss(reduction='none')(yp_next, y)
                log_loss_next = theta * torch.log(loss_next + 1e-10)

                log_ratio = log_loss_next - log_loss 

                idx_ = torch.where(log_ratio > np.log(1))
                log_ratio[idx_].fill_(np.log(1))
                u = torch.log(torch.zeros_like(log_ratio).uniform_(0,1)).cuda()
                idx_accept = torch.where(u <= log_ratio.cuda())[0]
                transforms[idx_accept] = new_transforms[idx_accept]
                previous.data[idx_accept] = proposal.data[idx_accept].half()

        adv_loss = losses.detach().clone()
        adv_loss = torch.max(adv_loss, dim=1)[0]
        adv_loss = adv_loss.mean()
        total_adv_loss += adv_loss * y.size(0)

        loss = losses.detach().clone()
        loss = torch.exp(torch.log(loss + 1e-10).sum(dim=1) / m)
        loss = loss.mean() 
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset), total_adv_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    args = parser.parse_args()

    d = torch.load(args.checkpoint)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False)
    test_data = test_dataset.data[:1000]
    test_labels = test_dataset.targets[:1000]
    test_examples = list(zip(transpose(pad(test_data, 4) / 255.), test_labels))
    batch_size = 100
    test_loader = Batches(test_examples, batch_size, shuffle=False, num_workers=2)

    model = preactresnet18().cuda()
    model.load_state_dict(d["model"])
    model.eval()
    _, test_loss = epoch_standard(test_loader, model)
    print('standard loss: ', test_loss)
    adv_losses = []
    for p in [1, 10, 100, 1000]:
        test_loss, adv_loss = epoch_mcmc(test_loader, model, p=p, m=500)
        print(f'mcmc loss p={p}: ', test_loss)
        test_loss = epoch_discrete_random_sampling(test_loader, model, p=p, m=500)
        print(f'rs loss p={p}: ', test_loss)
        adv_losses.append(adv_loss)
    print('adversarial loss: ', max(adv_losses))

if __name__ == '__main__':
    main()

