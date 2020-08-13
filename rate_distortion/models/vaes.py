# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from ..utils.vae_utils import log_normal, log_mean_exp, log_normal_likelihood
from ..utils.computation_utils import singleton_repeat
from .registry import register
from torch.nn import functional as F
from torch.distributions.normal import Normal


@register("blurry_vae")
def get_vae(hparams):
    """ 
    The deep blurry VAE model used in the experiments. 
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            # encoder
            self.en1 = nn.Linear(hparams.dataset.input_vector_length, 1024)
            self.en2 = nn.Linear(1024, 1024)
            self.en3 = nn.Linear(1024, 1024)
            self.en4 = nn.Linear(1024, hparams.model_train.z_size * 2)
            # decoder
            self.de1 = nn.Linear(hparams.model_train.z_size, 1024)
            self.de2 = nn.Linear(1024, 1024)
            self.de3 = nn.Linear(1024, 1024)
            self.de_mean = nn.Linear(1024, hparams.dataset.input_vector_length)
            self.x_logvar = nn.Parameter(
                torch.log(
                    torch.tensor(
                        hparams.model_train.x_var, dtype=hparams.tensor_type)),
                requires_grad=False)
            self.observation_log_likelihood_fn = log_normal_likelihood
            self.Guassian_Kernal = None
            self.hparams = hparams

        def encode(self, x):
            h1 = torch.tanh(self.en1(x))
            h2 = torch.tanh(self.en2(h1))
            h3 = torch.tanh(self.en3(h2))
            latent = self.en4(h3)
            mean, logvar = latent[:, :hparams.model_train.
                                  z_size], latent[:, hparams.model_train.
                                                  z_size:]
            return mean, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):

            h1 = torch.tanh(self.de1(z))
            h2 = torch.tanh(self.de2(h1))
            h3 = torch.tanh(self.de3(h2))

            mean = self.de_mean(h3)
            output_img = torch.sigmoid(mean).view(
                -1, self.hparams.dataset.input_dims[1],
                self.hparams.dataset.input_dims[2])
            output_img = torch.unsqueeze(output_img, 1)
            output_img = F.pad(output_img, (2, 2, 2, 2), mode='reflect')
            blurred_img = self.Guassian_Kernal(output_img)

            return blurred_img.view(
                -1, hparams.dataset.input_vector_length), self.x_logvar

        def forward(self, x, num_iwae=1):
            flattened_x = x.view(-1, hparams.dataset.input_vector_length)
            flattened_x_k = singleton_repeat(flattened_x, num_iwae)

            mu, logvar = self.encode(flattened_x_k)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)
            x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
            likelihood = self.observation_log_likelihood_fn(
                flattened_x_k, x_mean, x_logvar_full)
            elbo = likelihood + logpz - logqz

            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)


@register("deep_vae")
def get_vae(hparams):
    """ 
    The deep VAE model used in the experiments. 
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            # encoder
            self.en1 = nn.Linear(hparams.dataset.input_vector_length, 1024)
            self.en2 = nn.Linear(1024, 1024)
            self.en3 = nn.Linear(1024, 1024)
            self.en4 = nn.Linear(1024, hparams.model_train.z_size * 2)
            # decoder
            self.de1 = nn.Linear(hparams.model_train.z_size, 1024)
            self.de2 = nn.Linear(1024, 1024)
            self.de3 = nn.Linear(1024, 1024)
            self.de_mean = nn.Linear(1024, hparams.dataset.input_vector_length)
            self.x_logvar = nn.Parameter(
                torch.log(
                    torch.tensor(
                        hparams.model_train.x_var, dtype=hparams.tensor_type)),
                requires_grad=True)
            self.observation_log_likelihood_fn = log_normal_likelihood

        def encode(self, x):
            h1 = torch.tanh(self.en1(x))
            h2 = torch.tanh(self.en2(h1))
            h3 = torch.tanh(self.en3(h2))
            latent = self.en4(h3)
            mean, logvar = latent[:, :hparams.model_train.
                                  z_size], latent[:, hparams.model_train.
                                                  z_size:]
            return mean, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):

            h1 = torch.tanh(self.de1(z))
            h2 = torch.tanh(self.de2(h1))
            h3 = torch.tanh(self.de3(h2))

            mean = self.de_mean(h3)
            return torch.sigmoid(mean), self.x_logvar

        def forward(self, x, num_iwae=1):
            flattened_x = x.view(-1, hparams.dataset.input_vector_length)
            flattened_x_k = singleton_repeat(flattened_x, num_iwae)
            mu, logvar = self.encode(flattened_x_k)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)
            x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
            likelihood = self.observation_log_likelihood_fn(
                flattened_x_k, x_mean, x_logvar_full)
            elbo = likelihood + logpz - logqz

            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)


@register("vae_linear_fixed_var")
def get_vae(hparams):
    """ 
    The VAE model used for Analytical solution.
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            self.enc = nn.Linear(hparams.dataset.input_vector_length,
                                 hparams.model_train.z_size * 2)
            self.dec_mean = nn.Linear(hparams.model_train.z_size,
                                      hparams.dataset.input_vector_length)
            self.observation_log_likelihood_fn = log_normal_likelihood
            self.x_logvar = nn.Parameter(
                torch.log(torch.tensor(1, dtype=hparams.tensor_type)),
                requires_grad=True)

        def encode(self, x):
            hidden = self.enc(x)
            mean, logvar = hidden[:, :hparams.model_train.
                                  z_size], hidden[:, hparams.model_train.
                                                  z_size:]
            return mean, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            logqz = log_normal(z, mu, logvar)
            zeros = torch.zeros(z.size()).type(hparams.dtype)
            logpz = log_normal(z, zeros, zeros)
            return z, logpz, logqz

        def decode(self, z):
            h5 = self.dec_mean(z)
            return h5, torch.zeros(1)

        def forward(self, x, num_iwae=1):
            flattened_x = x.view(-1, hparams.dataset.input_vector_length)
            flattened_x_k = singleton_repeat(flattened_x, num_iwae)
            mu, logvar = self.encode(flattened_x_k)
            z, logpz, logqz = self.reparameterize(mu, logvar)
            x_mean, x_logvar = self.decode(z)

            likelihood = self.observation_log_likelihood_fn(
                flattened_x_k, x_mean, x_logvar)
            elbo = likelihood + logpz - logqz

            if num_iwae != 1:
                elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
                logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
                logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
                likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
            elbo = torch.mean(elbo)
            logpz = torch.mean(logpz)
            logqz = torch.mean(logqz)
            likelihood = torch.mean(likelihood)
            return x_mean, elbo, mu, logvar, -likelihood, logqz - logpz, z

    return VAE(hparams)
