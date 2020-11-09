# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/utils.py

import numpy as np
from math import pi as pi
import torch
from ..models.registry import get_model
from .experiment_utils import note_taking, load_checkpoint
from ..algorithms.train_vae import train_and_test
from .guassian_blur import GaussianSmoothing
from torch import nn
from torch import optim
import math
import numbers
from torch.nn import functional as F
from torch.distributions.normal import Normal


def prepare_vae(writer, hparams):
    """ 
    Prepare the VAE decoder. If hparams.train_first, it'll train first. Else load checkpoint.
    """
    model = get_model(hparams).to(hparams.device)

    if hparams.train_first or hparams.checkpoint_path is None:
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
        if hparams.resume_training:
            try:
                hparams.step, hparams.epoch, test_loss_list = load_checkpoint(
                    path=hparams.checkpoint_path,
                    optimizer=optimizer,
                    reset_optimizer=False,
                    model=model)
            except:
                note_taking("Failure loading checkpoint from: {}...".format(
                    hparams.checkpoint_path))
                note_taking(
                    "re-initializing... will train from scracth instead, in: {}"
                    .format(hparams.messenger.checkpoint_dir))
                hparams.epoch = 0
                hparams.step = 0
                test_loss_list = list()

        else:
            hparams.epoch = 0
            hparams.step = 0
            test_loss_list = list()
        model = train_and_test(test_loss_list, model, optimizer, writer,
                               hparams)

    else:
        hparams.step, hparams.epoch, _ = load_checkpoint(
            path=hparams.checkpoint_path
            if hparams.train_first else hparams.checkpoint_path,
            optimizer=None,
            reset_optimizer=False,
            model=model)

    if hparams.blur_std is not None:
        model.Guassian_Kernal = GaussianSmoothing(
            channels=1,
            kernel_size=5,
            sigma=hparams.blur_std,
            tensor_type=hparams.tensor_type)

    if hparams.overwrite_variance is not None:
        model.x_logvar = nn.Parameter(
            torch.log(
                torch.tensor(
                    hparams.overwrite_variance, dtype=hparams.tensor_type)),
            requires_grad=False)
        note_taking("NOTICE! model.x_logvar is overwrited: {}".format(
            model.x_logvar))
    return model


def log_normal_likelihood(x, mean, logvar):
    """Implementation WITH constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py

    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """

    dim = list(mean.size())[1]
    logvar = torch.zeros(mean.size()) + logvar
    return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(1) +
                   torch.log(torch.tensor(2 * pi)) * dim)


def log_mean_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(torch.mean(torch.exp(x - max_), dim)) + torch.squeeze(max_)


def log_normal(x, mean, logvar):
    """
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    log normal WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out later
    Args:s
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """
    return -0.5 * (logvar.sum(1) + (
        (x - mean).pow(2) / torch.exp(logvar)).sum(1))
