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
from utils import *
from models.preactresnet import preactresnet18
from models.mnist import mnist_classifier
from datasets import get_train_loader, get_test_loader
import torch.nn.functional as F


mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()


def normalize(X):
    return (X - mu)/std


def pgd_linf(model, X, y, epsilon=0.03, m=20, randomize=True, alpha_scale=2.5, restarts=1):
    """ Construct FGSM adversarial examples on the examples X"""
    alpha = alpha_scale * epsilon / m

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        if randomize:
            delta = torch.zeros_like(X)
            delta.uniform_(-epsilon, epsilon)
            delta.requires_grad = True
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        for t in range(m):
            if X.shape[1] == 1:
                loss = nn.CrossEntropyLoss()(model(torch.clamp(X + delta, min=0, max=1)), y)
            else:
                loss = nn.CrossEntropyLoss()(model(normalize(torch.clamp(X + delta, min=0, max=1))), y)
            loss.backward()
            delta.data = delta + alpha*delta.grad.detach().sign()
            delta.data = torch.clamp(delta, -epsilon, epsilon)
            delta.grad.zero_()
        if X.shape[1] == 1:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta.detach()


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


def epoch_adversarial(loader, model, opt=None, eval_mode=False, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for batch in loader:
        if type(batch) is dict:
            X, y = batch['input'], batch['target']
        else:
            X, y = batch
        X,y = X.cuda(), y.cuda()
        delta = pgd_linf(model, X, y, **kwargs)
        if X.shape[1] == 1:
            yp = model(X+delta)
        else:
            yp = model(normalize(X+delta))
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_random_sampling(loader, model, opt=None, epsilon=0.03, p=1, m=10, eval_mode=False):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss = 0.
    for batch in loader:
        if type(batch) is dict:
            X, y = batch['input'], batch['target']
        else:
            X, y = batch

        X,y = X.cuda(), y.cuda()
        lower_limit = torch.max(-X, torch.tensor(-epsilon, dtype=X.dtype).view(1, 1, 1).cuda())
        upper_limit = torch.min(1 - X, torch.tensor(epsilon, dtype=X.dtype).view(1, 1, 1).cuda())
        if not eval_mode:
            deltas = (lower_limit - upper_limit) * torch.rand(m, *X.shape).cuda() + upper_limit
            deltas.requires_grad = True

            X_delta = (X[None] + deltas).transpose(0, 1).contiguous().view(-1, *X.shape[1:])
            y_delta = y[None].expand(m,*y.shape).transpose(0, 1).contiguous().view(-1)
            if X.shape[1] == 1:
                yp_delta = model(X_delta)
            else:
                yp_delta = model(normalize(X_delta))
            loss = nn.CrossEntropyLoss(reduction='none')(yp_delta,y_delta)
            loss = loss.view(X.size(0), m)
            loss = (torch.exp(torch.logsumexp(torch.log(loss + 1e-10) * p - math.log(m), dim=1)/ p)).mean()
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            losses = torch.zeros(m, X.size(0)).cuda()
            for i in range(m):
                delta = (lower_limit - upper_limit) * torch.rand_like(X) + upper_limit
                if X.shape[1] == 1:
                    yp_delta = model(X + delta)
                else:
                    yp_delta = model(normalize(X + delta))
                loss = nn.CrossEntropyLoss(reduction='none')(yp_delta,y)
                losses[i] = loss.detach()
            loss = losses.transpose(0, 1)
            loss = (torch.exp(torch.logsumexp(torch.log(loss + 1e-10) * p - math.log(m), dim=1)/ p)).mean()
        total_loss += loss.item() * y.size(0)
    return 0.0, total_loss / len(loader.dataset)


def epoch_hmc(loader, model, opt=None, epsilon=0.03, p=1, m=10, l=1, path_len=0.05, sigma=0.1,
            eval_mode=False, anneal_theta=True):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss = 0.
    num_accepts = 0
    total_n = 0

    alpha = path_len * sigma ** 2 / l
    print(alpha)

    for batch in loader:
        if type(batch) is dict:
            X, y = batch['input'], batch['target']
        else:
            X, y = batch
        X,y = X.cuda(), y.cuda()

        lower_limit = torch.max(-X, torch.tensor(-epsilon, dtype=X.dtype).view(1, 1, 1).cuda())
        upper_limit = torch.min(1 - X, torch.tensor(epsilon, dtype=X.dtype).view(1, 1, 1).cuda())

        if eval_mode:
            losses = torch.zeros(X.size(0), m)
        else:
            deltas = torch.zeros(m, *X.shape).cuda()

        delta = (lower_limit - upper_limit) * torch.rand_like(X) + upper_limit
        delta.requires_grad = True

        if anneal_theta:
            thetas = np.linspace(0, p, m)
        else:
            thetas = [np.random.uniform(0, p) for i in range(m)]
        model.eval()
        for i, theta in enumerate(thetas):
            mom = torch.randn_like(X).cuda() * sigma
            if not eval_mode:
                deltas[i] = delta.data

            if X.shape[1] == 1:
                yp = model(X + delta)
            else:
                yp = model(normalize(X + delta))
            loss = nn.CrossEntropyLoss(reduction='none')(yp,y)
            if eval_mode:
                losses[:, i] = loss.detach().cpu()
            log_loss = theta * torch.log(loss + 1e-10)
            log_loss.sum().backward()

            h_delta = torch.norm(mom.view(X.size(0), -1), dim=1)**2/sigma**2/2 - log_loss
            mom += 0.5 * alpha * delta.grad # half step of momentum
            proposal = delta.data
            for j in range(l):
                proposal = proposal.data + alpha * mom / sigma**2    # full step of position
                # reflection
                while len(torch.where(proposal < lower_limit)[0]) > 0 or len(torch.where(proposal > upper_limit)[0]) > 0:
                    idx_ = torch.where(proposal < lower_limit)
                    if len(idx_[0]) > 0:
                        proposal.data[idx_] = 2*lower_limit[idx_] - proposal.data[idx_]
                        mom[idx_] = -mom[idx_]
                    idx_ = torch.where(proposal > upper_limit)
                    if len(idx_[0]) > 0:
                        proposal.data[idx_] = 2*upper_limit[idx_] - proposal.data[idx_]
                        mom[idx_] = -mom[idx_]
                proposal.requires_grad = True
                if X.shape[1] == 1:
                    yp_next = model(X + proposal)
                else:
                    yp_next = model(normalize(X + proposal))
                loss_next = nn.CrossEntropyLoss(reduction='none')(yp_next, y)
                log_loss_next = theta * torch.log(loss_next + 1e-10)
                log_loss_next.sum().backward()
                if j != (l-1):
                    mom += alpha * proposal.grad         # full step of momentum
            mom += 0.5 * alpha * proposal.grad

            h_proposal = torch.norm(mom.view(X.size(0), -1), dim=1)**2/sigma**2/2 - log_loss_next
            delta_h = h_proposal - h_delta
            u = torch.zeros_like(delta_h).uniform_(0,1)
            idx_accept = torch.where(u <= torch.exp(-delta_h))
            delta.data[idx_accept] = proposal.data[idx_accept]

            num_accepts += len(idx_accept[0])
            total_n += delta.size(0)
            delta.grad.zero_()

        if eval_mode:
            loss = losses
        else:
            model.train()
            y_delta = y[None].expand(m,*y.shape).transpose(0, 1).contiguous().view(-1)
            X_delta = (X[None] + deltas).transpose(0, 1).contiguous().view(-1, *X.shape[1:])
            if X.shape[1] == 1:
                yp_delta = model(torch.clamp(X_delta, min=0, max=1))
            else:
                yp_delta = model(normalize(torch.clamp(X_delta, min=0, max=1)))
            loss = nn.CrossEntropyLoss(reduction='none')(yp_delta,y_delta)
            loss = loss.view(X.size(0), m)
        loss = torch.exp(torch.log(loss + 1e-10).sum(dim=1) / m)
        loss = loss.mean()
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.item() * y.size(0)

    print('percent accepts ', num_accepts / total_n * 100.)
    return 0.0, total_loss / len(loader.dataset)


epoch_modes = {
    "standard": epoch_standard,
    "adversarial": epoch_adversarial,
    "hmc": epoch_hmc,
    "random_sampling": epoch_random_sampling
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
        if config.training.lr_schedule == 'cyclic':
            lr_schedule = lambda t: np.interp([t], [0, config.training.epochs * 2 // 5, config.training.epochs], [0, config.training.opt.params.lr, 0])[0]
        elif config.training.lr_schedule == 'multi_step':
            def lr_schedule(t):
                if t / config.training.epochs < 0.5:
                    return config.training.opt.params.lr
                elif t / config.training.epochs < 0.75:
                    return config.training.opt.params.lr / 10.
                else:
                    return config.training.opt.params.lr / 100.

    if config.data.dataset == 'mnist':
        model = mnist_classifier().cuda()
    elif config.data.dataset == 'cifar10':
        model = preactresnet18().cuda()

    if config.training is not None:
        if config.data.use_half:
            transforms = [Crop(32, 32), FlipLR()]
            dataset = cifar10(config.data.root)
            train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
            train_set_x = Transform(train_set, transforms)
            train_loader = Batches(train_set_x, config.data.training.batch_size, shuffle=True, set_random_choices=True, num_workers=config.data.num_workers)

            test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
            test_loader = Batches(test_set, config.data.test.batch_size, shuffle=False, num_workers=config.data.num_workers)
        else:
            train_loader = get_train_loader(config)
            test_loader = get_test_loader(config)
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
