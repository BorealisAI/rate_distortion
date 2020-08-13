# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from scipy.stats import norm
import scipy.linalg as linalg
from scipy.stats import multivariate_normal
import numpy as np
import torch
from ..data.registry import get_loader
from torchvision.utils import save_image
from ..utils.experiment_utils import note_taking, sample_images
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal


class Mixed_Gaussian(object):

    def __init__(self, Ber_dist, original_prior, bad_prior, hparams):
        self.Ber_dist = Ber_dist
        self.original_prior = original_prior
        self.bad_prior = bad_prior
        self.hparams = hparams

    def likelihood(self, x):
        x = torch.from_numpy(x).to(
            device=self.hparams.device, dtype=self.hparams.tensor_type)
        prob = self.hparams.mixture_weight * (torch.exp(
            self.bad_prior.log_prob(x))) + (
                1 - self.hparams.mixture_weight) * torch.exp(
                    self.original_prior.log_prob(x))
        return prob

    def log_likelihood(self, x):
        return torch.sum(
            torch.log(self.likelihood(x)), dim=1).detach().cpu().numpy()

    def sample(self, shape=[]):
        coin_flips = torch.squeeze(
            self.Ber_dist.sample([shape]).type(torch.ByteTensor))
        note_taking("coin_flips: {}".format(coin_flips))
        good_samples = self.original_prior.sample([shape])
        bad_samples = self.bad_prior.sample([shape])
        bad_samples_selected = bad_samples[coin_flips]
        good_samples_selected = good_samples[(1 - coin_flips)]
        samples = torch.cat((bad_samples_selected, good_samples_selected),
                            dim=0)
        return samples.detach().cpu().numpy()


def get_mix_prior(hparams, model):
    Ber_dist = Bernoulli(torch.tensor([hparams.mixture_weight]))
    mean_zeros = torch.zeros(hparams.model_train.z_size)
    std_ones = torch.ones(hparams.model_train.z_size)
    std_bad = torch.ones(hparams.model_train.z_size) * hparams.bad_std
    original_prior = Normal(loc=mean_zeros, scale=std_ones)
    bad_prior = Normal(loc=mean_zeros, scale=std_bad)
    mixture_prior = Mixed_Gaussian(Ber_dist, original_prior, bad_prior, hparams)
    samples_64 = torch.from_numpy(mixture_prior.sample(64)).to(
        device=hparams.device, dtype=hparams.tensor_type)
    sample_images(
        hparams,
        model,
        hparams.epoch,
        prior_dist=mixture_prior,
        name="agg_post",
        sample_z=samples_64)
    return mixture_prior
