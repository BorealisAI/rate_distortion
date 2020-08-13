# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/ais.py

#!/usr/bin/env python3

import numpy as np
import torch
from ..utils.vae_utils import log_normal_likelihood
from ..utils.experiment_utils import note_taking, save_comparison
_ANNOPS = dict()


def register(name):

    def add_to_dict(fn):
        global _ANNOPS
        _ANNOPS[name] = fn
        return fn

    return add_to_dict


def get_anneal_operators(target_dist):
    return _ANNOPS[target_dist]()


def distortion2ais(t, distortion):
    return (1. / torch.exp(distortion))**t


@register("joint_xz")
def joint_xz():
    """ 
    Annealing distribution for pixel NLL distortion metric. 
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist):
        """
        Compute unnormalized density for intermediate distribution:
        p_t = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        log p_t = log p(z) + t * log p(x|z)
        """
        z_zeros = torch.zeros(
            task_params.num_total_chains,
            hparams.model_train.z_size,
            dtype=hparams.tensor_type)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        x_mean, x_logvar = model.decode(z)
        x_logvar_full = torch.zeros(
            x_mean.size(), dtype=hparams.tensor_type) + x_logvar
        log_likelihood = log_normal_likelihood(data, x_mean, x_logvar_full)
        if hparams.messenger.save_post_images:
            save_comparison(
                data,
                x_mean,
                task_params.batch_size,
                hparams,
                beta=hparams.messenger.beta)
            hparams.messenger.save_post_images = False

        return log_prior + log_likelihood * t

    return anneal_dist


@register("MSE")
def MSE():
    """ 
    Annealing distribution for pixel MSE distortion metric. 
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist):
        """
        Compute unnormalized density for intermediate distribution:
        p_t = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        log p_t = log p(z) + t * log p(x|z)
        """
        z_zeros = torch.zeros(task_params.num_total_chains,
                              hparams.model_train.z_size)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        x_mean, _ = model.decode(z)
        MSE = torch.sum((data - x_mean)**2, dim=1)
        if hparams.messenger.save_post_images:
            save_comparison(
                data,
                x_mean,
                task_params.batch_size,
                hparams,
                beta=hparams.messenger.beta)
            hparams.messenger.save_post_images = False

        return log_prior - t * MSE

    return anneal_dist


@register("MNIST_fid")
def MSE():
    """ 
    Annealing distribution for deep features MSE distortion metric. 
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist,
                    rep_model):
        """
        Compute unnormalized density for intermediate distribution:
        p_t = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        log p_t = log p(z) + t * log p(x|z)
      """
        z_zeros = torch.zeros(task_params.num_total_chains,
                              hparams.model_train.z_size)
        log_prior = log_normal_likelihood(z, z_zeros, z_zeros)
        x_mean, _ = model.decode(z)
        gen_rep = rep_model.get_rep_2(x_mean)
        data_rep = rep_model.get_rep_2(data)
        MSE = torch.sum((gen_rep - data_rep)**2, dim=1)
        if hparams.messenger.save_post_images:
            save_comparison(
                data,
                x_mean,
                task_params.batch_size,
                hparams,
                beta=hparams.messenger.beta)
            hparams.messenger.save_post_images = False

        return log_prior - t * MSE

    return anneal_dist


@register("mix_prior_MSE")
def mix_prior():
    """ 
    Annealing distribution used for mixture of prior experiment. 
    This one gives the flexibility of passing any prior distribution into the computation.
    """

    def anneal_dist(z, data, t, model, hparams, task_params, prior_dist):
        """
        Compute unnormalized density for intermediate distribution:
        p_t = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        log p_t = log p(z) + t * log p(x|z)
        """
        log_prior = torch.from_numpy(
            prior_dist.log_likelihood(z.detach().cpu().numpy())).to(
                hparams.tensor_type).to(hparams.device)
        x_mean, _ = model.decode(z)
        MSE = torch.sum((data - x_mean)**2, dim=1)
        if hparams.messenger.save_post_images:
            save_comparison(
                data,
                x_mean,
                task_params.batch_size,
                hparams,
                beta=hparams.messenger.beta)
            hparams.messenger.save_post_images = False

        return log_prior - t * MSE

    return anneal_dist
