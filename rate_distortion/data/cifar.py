# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import register
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler


def init_new_CIFAR10_loader(batch_size,
                            data_dir,
                            train=False,
                            shuffle=False,
                            data="CIFAR10",
                            cuda=True,
                            normalize=None):

    if normalize is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((normalize[0], normalize[0], normalize[0]),
                                 (normalize[1], normalize[1], normalize[1]))
        ])
    else:
        transform = transforms.ToTensor()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_dir, train=train, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=(True if shuffle else False),
        **kwargs)
    return loader


@register("cifar_train_valid")
def cifar_train_valid(batch_size, hparams):
    kwargs = {'num_workers': 1, 'pin_memory': True} if hparams.cuda else {}

    if hparams.dataset.normalize is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (hparams.dataset.normalize[0], hparams.dataset.normalize[0],
                 hparams.dataset.normalize[0]),
                (hparams.dataset.normalize[1], hparams.dataset.normalize[1],
                 hparams.dataset.normalize[1]))
        ])
    else:
        transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        hparams.data_dir, train=True, download=True, transform=transform)
    valid_dataset = datasets.CIFAR10(
        hparams.data_dir, train=True, download=True, transform=transform)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    np.random.seed(hparams.random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler, **kwargs)
    return train_loader, valid_loader


@register("CIFAR10_eval_train")
def CIFAR10_eval_train(batch_size, hparams):
    """ 
    Use it for testing on the training set.
    Will load the entire training set.
    """
    return init_new_CIFAR10_loader(
        batch_size,
        hparams.data_dir,
        train=True,
        shuffle=False,
        data="CIFAR10",
        cuda=hparams.cuda,
        normalize=hparams.dataset.normalize)


@register("CIFAR10_eval_test")
@register("CIFAR10_test")
def CIFAR10_test(batch_size, hparams):
    """ 
    Test loader for CIFAR. 
    """
    return init_new_CIFAR10_loader(
        batch_size,
        hparams.data_dir,
        train=False,
        shuffle=False,
        data="CIFAR10",
        cuda=hparams.cuda,
        normalize=hparams.dataset.normalize)
