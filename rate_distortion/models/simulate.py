# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/simulate.py

import numpy as np
import torch
from torch.distributions import Bernoulli, Normal
from torchvision.utils import save_image
from ..utils.experiment_utils import note_taking

_SDATA = dict()
""" 
The simulated data is returned as list, instead of iterators. 
Justification: Need to run through it multiply times, and:
"This itertool may require significant auxiliary storage 
(depending on how much temporary data needs to be stored). 
In general, if one iterator uses most or all of the data 
before another iterator starts, it is faster to use list() 
instead of tee()."
https://docs.python.org/3/library/itertools.html#itertools.tee
 """


def register(name):

    def add_to_dict(fn):
        global _SDATA
        _SDATA[name] = fn
        return fn

    return add_to_dict


def get_simulate_data(model, batch_size, n_batch, hparams, rd=False, beta=None):
    return _SDATA[hparams.simulated_data](
        model, batch_size, n_batch, hparams, rd=False, beta=beta)


def save_simulated_data(hparams, x, x_mean, i, beta=None):
    if beta is None:
        mean_path = hparams.messenger.image_dir + 'simulated_mean_iteration_' + str(
            i) + '.png'
        x_path = hparams.messenger.image_dir + 'simulated_data_iteration_' + str(
            i) + '.png'
    else:
        mean_path = "{}_simulated_mean_beta_{}_iteration_{}.png".format(
            hparams.messenger.image_dir, beta, str(i))
        x_path = "{}_simulated_data_beta_{}_iteration_{}.png".format(
            hparams.messenger.image_dir, beta, str(i))
    save_image(x.view(-1, *hparams.dataset.input_dims), x_path)
    save_image(x_mean.view(-1, *hparams.dataset.input_dims), mean_path)


@register("simulate_default")
def simulate_data(model, batch_size, n_batch, hparams, rd=False, beta=None):
    """
    Simulate data for a decoder based generative model.
    Args:
        model: model for data simulation
        batch_size: batch size for simulated data
        n_batch: number of batches

    Returns:
        list of batches of torch Tensor pair x, z
    """

    gpu_batches = list()
    x_list = list()
    z_list = list()
    for i in range(n_batch):
        # assume prior is unit Gaussian
        z = torch.randn([batch_size,
                         hparams.model_train.z_size]).requires_grad_().to(
                             device=hparams.device, dtype=hparams.tensor_type)
        x_mean, x_logvar = model.decode(z)
        if beta is not None:
            if hparams.rd.target_dist == "joint_xz" or hparams.rd.target_dist == "mix_prior":
                x_logvar -= torch.log(torch.tensor(beta)).to(
                    device=hparams.device, dtype=hparams.tensor_type)
            else:
                x_logvar = -torch.log(torch.tensor(2 * beta)).to(
                    device=hparams.device, dtype=hparams.tensor_type)

        std = torch.ones(x_mean.size()).mul(torch.exp(x_logvar * 0.5))
        x_normal_dist = Normal(loc=x_mean, scale=std)
        x = x_normal_dist.sample().to(
            device=hparams.device, dtype=hparams.tensor_type)

        paired_batch = (x, z)
        gpu_batches.append(paired_batch)
        x_list.append(x)
        z_list.append(z)
        save_simulated_data(hparams, x, x_mean, i, beta)

    return gpu_batches
