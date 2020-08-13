# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import torch
import numpy as np
from ..utils.experiment_utils import note_taking, save_checkpoint, sample_images, get_chechpoint_path, load_checkpoint, init_dir, plot_rate_distrotion
from ..utils.computation_utils import singleton_repeat
from ..utils.vae_utils import log_normal, log_mean_exp, log_normal_likelihood
from torchvision.utils import save_image
import torch.nn as nn
from ..data.load_data import *
from tqdm import tqdm
import os
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import gc


def log_likelihood_torch(x, mean, logvar):
    normal_dist = Normal(mean, torch.exp(0.5 * logvar))
    return torch.sum(normal_dist.log_prob(x), dim=1)


class VAE(nn.Module):

    def __init__(self, hparams, decode, beta):
        super(VAE, self).__init__()
        # encoder
        self.en1 = nn.Linear(hparams.dataset.input_vector_length, 1024)
        self.en2 = nn.Linear(1024, 1024)
        self.en3 = nn.Linear(1024, 1024)
        self.en4 = nn.Linear(1024, hparams.model_train.z_size * 2)
        self.decode = decode
        self.hparams = hparams
        self.beta = beta
        self.observation_log_likelihood_fn = log_normal_likelihood
        if hparams.torch_likelihood:
            self.observation_log_likelihood_fn = log_likelihood_torch

    def encode(self, x):
        h1 = torch.tanh(self.en1(x))
        h2 = torch.tanh(self.en2(h1))
        h3 = torch.tanh(self.en3(h2))
        latent = self.en4(h3)
        mean, logvar = latent[:, :self.hparams.model_train.
                              z_size], latent[:, self.hparams.model_train.
                                              z_size:]
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        logqz = log_normal(z, mu, logvar)
        zeros = torch.zeros(z.size()).type(self.hparams.dtype)
        logpz = log_normal(z, zeros, zeros)
        return z, logpz, logqz

    def update_beta(self, beta):
        self.beta = beta

    def forward(self, x, num_iwae=1):
        flattened_x = x.view(-1, self.hparams.dataset.input_vector_length)
        flattened_x_k = singleton_repeat(flattened_x, num_iwae)

        mu, logvar = self.encode(flattened_x_k)
        z, logpz, logqz = self.reparameterize(mu, logvar)
        x_mean, x_logvar = self.decode(z)
        x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
        likelihood = self.observation_log_likelihood_fn(flattened_x_k, x_mean,
                                                        x_logvar_full)
        loss = self.beta * likelihood + (logpz - logqz)

        loss = torch.mean(loss)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        likelihood = torch.mean(likelihood)
        return x_mean, loss, mu, logvar, -likelihood, logqz - logpz, z


def freeze_decoder(decoder):
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder


def plot_variational_post(mean, covariance, beta, hparams, label="baseline"):
    mu = mean[0]
    cov = torch.tensor([[torch.exp(covariance[0][0]), 0.],
                        [0., torch.exp(covariance[0][1])]])
    note_taking("variational q mu at beta={} is {}".format(
        beta,
        mu.detach().cpu().numpy()))
    note_taking("variational q cov at beta={} is {}".format(
        beta,
        cov.detach().cpu().numpy()))
    m = MultivariateNormal(mu, cov)
    nbins = 300
    x = np.linspace(-2, 2, nbins)
    y = np.linspace(-2, 2, nbins)
    x_grid, y_grid = np.meshgrid(
        np.linspace(-2, 2, nbins), np.linspace(-2, 2, nbins))
    samples_grid = np.vstack([x_grid.flatten(), y_grid.flatten()]).transpose()
    density = torch.exp(
        m.log_prob(
            torch.from_numpy(samples_grid).to(hparams.tensor_type).to(
                hparams.device)))
    plt.contourf(
        x_grid,
        y_grid,
        density.view((nbins, nbins)).detach().cpu().numpy(),
        20,
        cmap='jet')
    plt.colorbar()
    path = hparams.messenger.arxiv_dir + hparams.hparam_set + "_{}_var_q_b{}.pdf".format(
        label, beta)
    plt.savefig(path)
    plt.close()


def compute_rate_distortion_baseline(data_loader, hparams, beta, b_VAE):
    for i, (batch, _) in enumerate(data_loader, 0):
        batch = batch.to(device=hparams.device, dtype=hparams.tensor_type)
        num_iwae = hparams.baseline_samples if hparams.baseline_samples is not None else 1
        x_mean, loss, mu, logvar, D, R, approx_zs = b_VAE(
            batch, num_iwae=num_iwae)
        if hparams.rd.target_dist != "joint_xz":
            flattened_x = batch.view(-1, hparams.dataset.input_vector_length)
            if num_iwae != 1:
                flattened_x = singleton_repeat(flattened_x, num_iwae)
            D = torch.mean(torch.sum((flattened_x - x_mean)**2, dim=1))
        if hparams.analytical_rate:
            R = (1. / (hparams.rd.batch_size * num_iwae)
                ) * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        image_path = hparams.messenger.image_dir + "beta{}_baseline.png".format(
            beta)
        if i == hparams.rd.n_batch - 1:
            break

    flattened_data = batch.view(-1, hparams.dataset.input_vector_length)
    data_comparison = singleton_repeat(flattened_data, num_iwae)
    data_path = hparams.messenger.image_dir + "baseline_original_sample.png"
    save_image(
        data_comparison.view(-1, hparams.dataset.input_dims[0],
                             hparams.dataset.input_dims[1],
                             hparams.dataset.input_dims[2])[:64],
        data_path,
        nrow=8)
    save_image(
        x_mean.view(-1, hparams.dataset.input_dims[0],
                    hparams.dataset.input_dims[1],
                    hparams.dataset.input_dims[2])[:64],
        image_path,
        nrow=8)
    if hparams.model_train.z_size == 2:
        plot_variational_post(mu, logvar, beta, hparams)

    return R, D


def run_rd_baseline(decoder, data_loader, hparams):
    decoder = freeze_decoder(decoder)
    beta_list = hparams.messenger.beta_list
    KLD_list = list()
    distortion_list = list()
    betas_done = list()

    for beta in beta_list:
        if hparams.verbose:
            note_taking("Running rd-baseline with beta value {}".format(beta))
        if (not hparams.baseline_reuse) or (beta == beta_list[0]):
            b_VAE = VAE(hparams, decoder.decode, beta)
        else:
            b_VAE.update_beta(beta)
        b_VAE = train_and_test_bvae(b_VAE, hparams, beta)
        with torch.no_grad():
            R, D = compute_rate_distortion_baseline(data_loader, hparams, beta,
                                                    b_VAE)
            KLD_list.append(R.detach().cpu().numpy())
            distortion_list.append(D.detach().cpu().numpy())
            betas_done.append(beta)
    plot_rate_distrotion(
        hparams,
        betas_done,
        KLD_list,
        distortion_list,
        "test",
        metric="baseline")
    return KLD_list, distortion_list


def train_epoch(epoch, model, train_loader, hparams, optimizer, task_params):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader, 1):
        data = data.to(device=hparams.device, dtype=hparams.tensor_type)
        optimizer.zero_grad()
        _, elbo, _, _, _, _, _ = model(data)
        loss = -elbo
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    note_taking(
        'Training epoch: {} Average negative elbo on train set: {}'.format(
            epoch, train_loss / batch_idx))

    try:
        num_samples = len(train_loader.dataset)
    except:
        num_samples = len(train_loader) * task_params.batch_size
    return train_loss, num_samples


def test_epoch(epoch, model, test_loader, hparams, task_params, do_logging,
               beta):
    model.eval()
    elbos = list()
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader, 1):
            data = data.to(device=hparams.device, dtype=hparams.tensor_type)
            recon_batch, elbo, _, _, _, _, _ = model(data)
            elbos.append(elbo.cpu().numpy().item())
            if do_logging:
                if epoch is not None:
                    if i == 1:
                        if hparams.dataset.data_name == "cifar10":
                            recon_batch = recon_batch / 2 + 0.5
                        n = min(data.size(0), 8)
                        comparison = torch.cat([
                            data[:n],
                            recon_batch.view(task_params.batch_size,
                                             hparams.dataset.input_dims[0],
                                             hparams.dataset.input_dims[1],
                                             hparams.dataset.input_dims[2])[:n]
                        ])
                        save_image(
                            comparison.cpu(),
                            hparams.messenger.image_dir +
                            'beta{}_reconstruction_'.format(beta) + str(epoch) +
                            '.png',
                            nrow=n)
            if i == hparams.n_test_batch:
                break

    test_loss = -np.mean(elbos)
    note_taking('Test loss: {} over {} batches'.format(test_loss, float(i)))
    return test_loss


def train_and_test_bvae(model, hparams, beta):
    train_loader, test_loader = load_training_data(hparams)
    bvae_checkpoint_dir = hparams.messenger.arxiv_dir + "checkpoint_betas/"
    init_dir(bvae_checkpoint_dir)
    checkpoint_path = os.path.join(bvae_checkpoint_dir,
                                   "best_beta{}.pth".format(beta))
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    if hparams.check_baseline:
        for name, param in model.named_parameters():
            note_taking("param name={}".format(name))
            note_taking("param.requires_grad? {}".format(param.requires_grad))

    test_loss_list = list()
    hparams.epoch = 0
    hparams.step = 0

    for epoch in tqdm(range(hparams.model_train.epochs)):
        train_loss, steps = train_epoch(epoch, model, train_loader, hparams,
                                        optimizer, hparams.model_train)
        hparams.step += steps
        hparams.epoch += 1
        with torch.no_grad():
            test_loss = test_epoch(
                epoch,
                model,
                test_loader,
                hparams,
                hparams.model_train,
                do_logging=False,
                beta=beta)
            test_loss_list.append(test_loss)
            if len(test_loss_list) > 1:
                if test_loss <= min(test_loss_list[:-1]):
                    note_taking('new min test loss is {} at epoch {}'.format(
                        test_loss, epoch))

                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        optimizer=optimizer,
                        save_optimizer_state=True,
                        model=model,
                        hparams=hparams,
                        test_loss_list=test_loss_list)
            else:
                note_taking('first checkpoint with test loss {} saved'.format(
                    test_loss))
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    optimizer=optimizer,
                    save_optimizer_state=True,
                    model=model,
                    hparams=hparams,
                    test_loss_list=test_loss_list)

    hparams.step, hparams.epoch, _ = load_checkpoint(
        path=checkpoint_path,
        optimizer=None,
        reset_optimizer=False,
        model=model)
    test_epoch(
        hparams.epoch,
        model,
        test_loader,
        hparams,
        hparams.model_train,
        do_logging=True,
        beta=beta)
    torch.cuda.empty_cache()
    gc.collect()
    return model
