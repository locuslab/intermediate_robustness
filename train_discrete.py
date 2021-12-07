import argparse
import json
import logging
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from utils import *
from models.preactresnet import preactresnet18
from datasets import get_train_loader, get_test_loader

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


def normalize(X):
    return (X - mu)/std


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


def epoch_discrete_random_sampling(loader, model, opt=None, p=1, m=20, eval_mode=False):
    padded = 40
    unpadded = 32
    total_loss = 0.
    for batch in loader:
        X, y = batch['input'], batch['target']
        X, y = X.cuda(), y.cuda()
        if eval_mode:
            losses = torch.zeros(m, X.size(0)).cuda()
        else:
            Xs = torch.zeros(m, X.size(0), X.size(1), unpadded, unpadded).cuda()
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
            if eval_mode:
                yp = model(normalize(sample))
                loss = nn.CrossEntropyLoss(reduction='none')(yp, y)
                losses[i] = loss.detach()
            else:
                Xs[i] = sample
        
        if eval_mode:
            loss = losses.transpose(0, 1)
            loss = (torch.exp(torch.logsumexp(torch.log(loss + 1e-10) * p, dim=1)/ p) * (1/m)**(1/p))
            loss = loss.mean()
        else:
            y_samples = y[None].expand(m,*y.shape).transpose(0, 1).contiguous().view(-1)
            X_samples = Xs.transpose(0, 1).contiguous().view(-1, X.shape[1], unpadded, unpadded)
            yp_samples = model(normalize(X_samples))
            loss = nn.CrossEntropyLoss(reduction='none')(yp_samples,y_samples)
            loss = loss.view(X.size(0), m)
            loss = (torch.exp(torch.logsumexp(torch.log(loss + 1e-10) * p, dim=1)/ p) * (1/m)**(1/p)).mean()
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
        total_loss += loss.item() * y.size(0)
    return 0, total_loss / len(loader.dataset)    


epoch_modes = {
    "standard": epoch_standard,
    "discrete_random_sampling": epoch_discrete_random_sampling
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file', required=True)
    parser.add_argument('--resume', default=None, help='epoch')
    parser.add_argument('--dp', action='store_true')
    args = parser.parse_args()
    config_dict = get_config(args.config)

    assert os.path.splitext(os.path.basename(args.config))[0] == config_dict['output_dir']

    torch.manual_seed(config_dict['seed'])
    torch.cuda.manual_seed(config_dict['seed'])

    if config_dict['training'] is None:
        output_dir = os.path.join('evaluations', config_dict['data']['dataset'], config_dict['output_dir'])
    else:
        output_dir = os.path.join('experiments', config_dict['data']['dataset'], config_dict['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = config_to_namedtuple(config_dict)

    logger = get_logger(__name__, output_dir)

    logger.info(f'cuda {torch.cuda.is_available()}')

    if config.training is not None:
        lr_schedule = lambda t: np.interp([t], [0, config.training.epochs * 2 // 5, config.training.epochs], [0, config.training.opt.params.lr, 0])[0]

    model = preactresnet18().cuda()

    if config.training is not None:
        dataset = cifar10(config.data.root)
        train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
        if config.training.type == 'discrete_random_sampling':
            train_loader = Batches(train_set, config.data.training.batch_size, shuffle=True, num_workers=config.data.num_workers)
            test_set = list(zip(transpose(pad(dataset['test']['data'], 4)/255.), dataset['test']['labels']))
            test_loader = Batches(test_set, config.data.test.batch_size, shuffle=False, num_workers=config.data.num_workers)
        else:
            transforms = [Crop(32, 32), FlipLR()]
            train_set_x = Transform(train_set, transforms)
            train_loader = Batches(train_set_x, config.data.training.batch_size, shuffle=True, set_random_choices=True, num_workers=config.data.num_workers)

            test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
            test_loader = Batches(test_set, config.data.test.batch_size, shuffle=False, num_workers=config.data.num_workers)
        if config.training.opt.type == 'adam':
            opt = torch.optim.Adam(model.parameters(), **config.training.opt.params._asdict())
        elif config.training.opt.type == 'sgd':
            opt = torch.optim.SGD(model.parameters(),**config.training.opt.params._asdict())

        if args.resume is not None:
            checkpoint_filename = os.path.join(output_dir, 'checkpoints', f'checkpoint_{args.resume}.pth')
            d = torch.load(checkpoint_filename)
            logger.info(f"Resume model checkpoint {d['epoch']}...")
            model.load_state_dict(d["model"])
            opt.load_state_dict(d["opt"])
            start_epoch = d["epoch"] + 1
        else:
            start_epoch = 0

        if args.dp:
            model = nn.DataParallel(model)

        # Train
        logger.info(f"Epoch \t \t Train Loss \t Train Error \t LR")
        for epoch in range(start_epoch, config.training.epochs):
            lr = lr_schedule(epoch)

            opt.param_groups[0]['lr'] = lr
            epoch_mode = epoch_modes[config.training.type]
            train_err, train_loss = epoch_modes[config.training.type](train_loader, model, opt, **config.training.params._asdict())
            logger.info(f'{epoch} \t \t \t {train_loss:.6f} \t \t {train_err:.6f} \t {lr:.6f}')

            # evaluate
            if (epoch+1) % config.training.test_interval == 0 or epoch + 1 == config.training.epochs:
                model.eval()
                for evaluation in config.evaluations:
                    test_err, test_loss = epoch_modes[evaluation.type](test_loader, model, **evaluation.params._asdict(), eval_mode=True)
                    logger.info(f"{evaluation.type}: \t Test Loss {test_loss:.6f} \t Test Error {test_err:.6f} \t {lr:.6f}")
                model.train()
            if (epoch+1) % config.training.checkpoint_interval == 0 or epoch + 1 == config.training.epochs:
                if args.dp:
                    save_m = model.module
                else:
                    save_m = model
                d = {"epoch": epoch, "model": save_m.state_dict(), "opt": opt.state_dict()}
                torch.save(d, os.path.join(output_dir, 'checkpoints', f'checkpoint_{epoch}.pth'))

    else:
        if config.data.use_half:
            dataset = cifar10(config.data.root)
            test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
            test_loader = Batches(test_set, config.data.test.batch_size, shuffle=False, num_workers=config.data.num_workers)
        else:
            test_loader = get_test_loader(config)
        checkpoint_filename = config.checkpoint_filename
        d = torch.load(checkpoint_filename)
        logger.info(f"Loading model checkpoint {config.checkpoint_filename}...")
        model.load_state_dict(d["model"])
        if args.dp:
            model = nn.DataParallel(model)
        model.eval()
        for evaluation in config.evaluations:
            test_err, test_loss = epoch_modes[evaluation.type](test_loader, model, eval_mode=True, **evaluation.params._asdict())
            logger.info(f"{evaluation.type}: \t Test Loss {test_loss:.6f} \t Test Error {test_err:.6f}")


if __name__ == "__main__":
    main()
