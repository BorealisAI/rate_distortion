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
from ..utils.experiment_utils import note_taking


def init_new_mnist(batch_size,
                   data_dir,
                   train=False,
                   shuffle=False,
                   data="MNIST",
                   cuda=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=train,
            download=True,
            transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=(True if shuffle else False),
        **kwargs)
    return loader


@register("mnist_train_valid")
def mnist_train_valid(batch_size, hparams):
    kwargs = {'num_workers': 1, 'pin_memory': True} if hparams.cuda else {}
    train_dataset = datasets.MNIST(
        hparams.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor())
    valid_dataset = datasets.MNIST(
        hparams.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor())
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = 10000
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


@register("mnist_eval_train")
def mnist_eval_train(batch_size, hparams):
    return init_new_mnist(
        batch_size,
        hparams.data_dir,
        train=True,
        shuffle=False,
        data="MNIST",
        cuda=hparams.cuda)


@register("mnist_eval_test")
@register("mnist_test")
def mnist_test(batch_size, hparams):
    """ 
    This loader is for evaluating RD. It supports loading only a particular digit class, 
    by spesifying hparams.dataset.label.
     """
    if hparams.dataset.label is None:
        return init_new_mnist(
            batch_size,
            hparams.data_dir,
            train=False,
            shuffle=False,
            data="MNIST",
            cuda=hparams.cuda)
    else:
        full_test_loader = init_new_mnist(
            batch_size,
            hparams.data_dir,
            train=False,
            shuffle=False,
            data="MNIST",
            cuda=hparams.cuda)

        mask = torch.ones(batch_size) * hparams.dataset.label
        mask = mask.to(device=hparams.device, dtype=hparams.tensor_type)
        for i, (batch, labels) in enumerate(full_test_loader, 0):
            labels = labels.to(hparams.tensor_type).to(hparams.device)
            mask = torch.eq(labels, mask)
            target_data = batch[mask]
            save_image(
                target_data.view(-1, hparams.dataset.input_dims[0],
                                 hparams.dataset.input_dims[1],
                                 hparams.dataset.input_dims[2]),
                hparams.messenger.image_dir + "original_label{}.png".format(
                    hparams.dataset.label),
            )
            break
        target_data = target_data.to(
            device=hparams.device, dtype=hparams.tensor_type)
        if hparams.messenger.running_rd:
            hparams.rd.batch_size = target_data.size()[0]
            hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains
        note_taking(
            "Loaded category data for MNIST {} digit class of size {}".format(
                hparams.dataset.label, target_data.size()))
        return [(target_data, hparams.dataset.label)]


@register("mnist_test_rand")
def mnist_test(batch_size, hparams):
    """ 
    This loader is for evaluating RD. It supports loading only a particular digit class, 
    by spesifying hparams.dataset.label.
     """
    if hparams.dataset.label is None:
        return init_new_mnist(
            batch_size,
            hparams.data_dir,
            train=False,
            shuffle=True,
            data="MNIST",
            cuda=hparams.cuda)
    else:
        full_test_loader = init_new_mnist(
            batch_size,
            hparams.data_dir,
            train=False,
            shuffle=True,
            data="MNIST",
            cuda=hparams.cuda)

        mask = torch.ones(batch_size) * hparams.dataset.label
        mask = mask.to(device=hparams.device, dtype=hparams.tensor_type)
        for i, (batch, labels) in enumerate(full_test_loader, 0):
            labels = labels.to(hparams.tensor_type).to(hparams.device)
            mask = torch.eq(labels, mask)
            target_data = batch[mask]
            save_image(
                target_data.view(-1, hparams.dataset.input_dims[0],
                                 hparams.dataset.input_dims[1],
                                 hparams.dataset.input_dims[2]),
                hparams.messenger.image_dir + "original_label{}.png".format(
                    hparams.dataset.label),
            )
            break
        target_data = target_data.to(
            device=hparams.device, dtype=hparams.tensor_type)
        if hparams.messenger.running_rd:
            hparams.rd.batch_size = target_data.size()[0]
            hparams.rd.num_total_chains = hparams.rd.batch_size * hparams.rd.n_chains
        note_taking(
            "Loaded category data for MNIST {} digit class of size {}".format(
                hparams.dataset.label, target_data.size()))
        return [(target_data, hparams.dataset.label)]
