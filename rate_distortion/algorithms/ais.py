# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/ais.py
# and the R code in R. Neil's paper https://arxiv.org/pdf/1206.1901.pdf

import numpy as np
import time
import gc
import torch
import os
from ..utils.vae_utils import log_mean_exp, log_normal_likelihood
from ..utils.computation_utils import singleton_repeat
from ..utils.experiment_utils import note_taking, init_dir
from .hmc import hmc_trajectory, accept_reject
from tqdm import tqdm
from .anneal_ops import get_anneal_operators
from torchvision.utils import save_image
from scipy import stats


class AIS_core(object):

    def __init__(self, target_dist, prior_dist, model, task_params, hparams,
                 rep_model):
        self.target_dist = target_dist
        self.model = model
        self.task_params = task_params
        self.hparams = hparams
        self.prior_dist = prior_dist
        self.anneal_dist_lookup = get_anneal_operators(task_params.target_dist)
        self.rep_model = rep_model

    def anneal_dist(self, z, data, t):
        if "fid" in self.task_params.target_dist:
            return self.anneal_dist_lookup(z, data, t, self.model, self.hparams,
                                           self.task_params, self.prior_dist,
                                           self.rep_model)
        else:
            return self.anneal_dist_lookup(z, data, t, self.model, self.hparams,
                                           self.task_params, self.prior_dist)

    def U(self, z, batch, t):
        return -self.anneal_dist(z, batch, t)

    def grad_U(self, z, batch, t):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(self.task_params.num_total_chains)
        U_temp = self.U(z, batch, t)
        grad = torch.autograd.grad(U_temp, z, grad_outputs=grad_outputs)[0]
        # clip by norm to avoid numerical instability
        grad = torch.clamp(
            grad, -self.task_params.num_total_chains *
            self.hparams.model_train.z_size * 100,
            self.task_params.num_total_chains * self.hparams.model_train.z_size
            * 100)
        grad.requires_grad_()
        return grad

    def normalized_kinetic(self, v):
        zeros = torch.zeros(self.task_params.num_total_chains,
                            self.hparams.model_train.z_size)
        return -log_normal_likelihood(v, zeros, zeros)
        # note: can change to log likelihood without constant


def run_ais_chain(model,
                  loader,
                  mode,
                  schedule,
                  hparams,
                  task_params,
                  start_state=None,
                  init_weights=None,
                  init_step_size=None,
                  init_history=None,
                  init_traj_length=0,
                  step_sizes_dir=None,
                  prior_dist=None,
                  rep_model=None):
    """Compute annealed importance sampling trajectories for a batch of data
    Args:
        
        model : A trained modol in pytorch, ideally pytorch 1.0.
        loader (iterator): iterator or list that returns pairs, with first component being `x`,
            second being z.
        mode (string): run forward or backward chain
        schedule (list or 1D np.ndarray): temperature schedule of the geometric annealling distributions.
            foward chain has increasing values, whereas backward has decreasing values
        task_params: specific hyper parameters for this task, just in 
            case some parameter are different for different tasks
        start_state: the initial z's. If starting from prior, then set this to None
        init_weights the initial AIS weights(in log scale). If starting from prior, then set this to None
        init_step_size: initial step sizes. If starting from prior, then set this to None
        init_history: initial acceptence history. If starting from prior, then set this to None 
        init_traj_length: The current AIS step finished in total. If starting from prior, then set this to 0
        prior_dist: prior distribution. If it's a standard unit Guassian, set this to None. 
            Otherwise please refer to distribution object Mixed_Gaussian defined in mixture_prior_rd.py 
 
    
    Returns:
        logws:A list of tensors where each tensor contains the log importance weights
        for the given batch of data. Note that weights of independent chains of a datapoint is averaged out. 
        approx_post_zs: The samples at the end of AIS chain
        data_loader: The data acutally used in this AIS run
        importance_weights: log importance weights for the given batch of data before averaging out the independnt chains
        epsilon: The step sizes 
        accept_hist: The accepence history of HMC
        current_traj_length: A counter, counting how many steps in total have been run. 
    """

    anneal_ops = AIS_core(task_params.target_dist, prior_dist, model,
                          task_params, hparams, rep_model)
    _time = time.time()
    logws = list()  # for output

    importance_weights = list()  #for importance weight adjusting

    if mode == 'forward':
        approx_post_zs = list()
        if not hparams.messenger.fix_data_simulation:
            data_loader = list()
        else:
            data_loader = loader
    note_taking('=' * 80)
    note_taking('In {} mode'.format(mode).center(80))

    for i, (batch, post_z) in enumerate(loader, 0):

        if not hparams.messenger.fix_data_simulation:
            flattened_batch = batch.view(
                -1, hparams.dataset.input_vector_length).to(
                    device=hparams.device, dtype=hparams.tensor_type)
            batch = singleton_repeat(flattened_batch, task_params.n_chains)

        if init_step_size is None:
            epsilon = torch.ones(task_params.num_total_chains) * 0.01
        else:
            epsilon = init_step_size

        # accept/reject history for tuning step size
        if init_history is None:
            accept_hist = torch.zeros(task_params.num_total_chains)
        else:
            accept_hist = init_history

        if init_weights is None:
            logw = torch.zeros(task_params.num_total_chains)
        else:
            logw = init_weights[i]
        if mode == 'forward':
            if start_state is None:
                if prior_dist is not None:
                    current_z = prior_dist.sample(task_params.num_total_chains)
                    current_z = torch.from_numpy(current_z).to(
                        hparams.tensor_type).to(
                            hparams.device).requires_grad_()
                else:
                    current_z = torch.randn(
                        task_params.num_total_chains,
                        hparams.model_train.z_size).requires_grad_()
            else:
                current_z = start_state[i]
                current_z.requires_grad_()
        else:
            current_z = singleton_repeat(post_z,
                                         task_params.n_chains).requires_grad_()
        next_z = None
        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            current_traj_length = j + init_traj_length
            if step_sizes_dir is not None:
                step_sizes_path = step_sizes_dir + str(
                    current_traj_length) + ".pt"
            else:
                note_taking("WARNING! Emtpy step size dir. ")

            # overwrite step size if there's a target to load
            if hparams.step_sizes_target is not None:
                epsilon = torch.load(step_sizes_path)

            if j == len(schedule) - 1:
                if i == 0:
                    hparams.messenger.save_post_images = True
            else:
                hparams.messenger.save_post_images = False

            if next_z is not None:
                current_z = next_z
                current_z.requires_grad_()
            with torch.no_grad():
                log_f_denominator = anneal_ops.anneal_dist(current_z, batch, t0)
                log_f_numerator = anneal_ops.anneal_dist(current_z, batch, t1)
                logw += (log_f_numerator - log_f_denominator)

            current_v = torch.randn(current_z.size())
            hmc_z, hmc_v = hmc_trajectory(current_z, batch, t1, current_v,
                                          anneal_ops, epsilon, hparams,
                                          task_params)

            # accept-reject step
            with torch.no_grad():
                next_z, epsilon, accept_hist = accept_reject(
                    current_z, batch, t1, current_v, hmc_z, hmc_v, epsilon,
                    accept_hist, current_traj_length, anneal_ops, hparams,
                    task_params)

                if hparams.step_sizes_target is None:
                    torch.save(epsilon, step_sizes_path)

            # If this is enabled, the mean and variance of zs will be tracked
            # And a normality test will be performed
            if hparams.verify_z:
                z_variance = torch.var(next_z, dim=1)
                z_mean = torch.mean(next_z, dim=1)
                z_numpy = next_z.cpu().numpy()
                _, p = stats.normaltest(z_numpy, axis=1)
                note_taking(
                    "\n t={}. Verifying z stats, mean={}, var={}".format(
                        t0, z_mean, z_variance))
                note_taking("p value={}".format(p))

            torch.cuda.empty_cache()
            gc.collect()

        if mode == "forward":
            if hparams.plot_rd_curve:
                importance_weights.append(logw)

        # average out independent chains，use log_mean_exp for numerical stability
        logw = log_mean_exp(logw.view(task_params.n_chains, -1), dim=0)

        if mode == 'backward':
            logw = -logw
        else:
            approx_post_zs.append(next_z)
            if not hparams.messenger.fix_data_simulation:
                data_loader.append((batch, post_z))

        logws.append(logw)

        _time = time.time()

        if i == hparams.rd.n_batch - 1:
            break

    if mode == 'forward':

        return logws, approx_post_zs, data_loader, importance_weights, epsilon, accept_hist, current_traj_length
    else:
        return logws


def run_ais_forward(model,
                    loader,
                    hparams,
                    data,
                    forward_schedule,
                    task_params,
                    start_state=None,
                    init_weights=None,
                    init_step_size=None,
                    init_history=None,
                    init_traj_length=0,
                    prior_dist=None,
                    rep_model=None):
    """ 
    A helper function that organize the AIS weights
    Args:
        model : A trained modol in pytorch, ideally pytorch 1.0.
        loader (iterator): iterator or list that returns pairs, with first component being `x`,
            second being z.
        data: which dataset is AIS being runing on. Train, test or simulate
        forward_schedule: temperature schedule of the geometric annealling distributions.
            foward chain has increasing values
        task_params: specific hyper parameters for this task, just in 
            case some parameter are different for different tasks
        start_state: the initial z's. If starting from prior, then set this to None
        init_weights the initial AIS weights(in log scale). If starting from prior, then set this to None
        init_step_size: initial step sizes. If starting from prior, then set this to None
        init_history: initial acceptence history. If starting from prior, then set this to None 
        init_traj_length: The current AIS step finished in total. If starting from prior, then set this to 0
        prior_dist: prior distribution. If it's a standard unit Guassian, set this to None. 
            Otherwise please refer to distribution object Mixed_Gaussian defined in mixture_prior_rd.py 
 

    Returns:
        logws:A list of tensors where each tensor contains the log importance weights
        for the given batch of data. Note that weights of independent chains of a datapoint is averaged out. 
        approx_post_zs: The samples at the end of AIS chain
        data_loader: The data acutally used in this AIS run
        importance_weights: log importance weights for the given batch of data before averaging out the independnt chains
        epsilon: The step sizes 
        accept_hist: The accepence history of HMC
        current_traj_length: A counter, counting how many steps in total have been run. 
    

    """
    step_sizes_path = hparams.messenger.step_sizes_dir + (
        (data + "_") if data is not None else '')

    forward_logws, approx_post_zs, simulated_data, AIS_weights, epsilon, accept_hist, current_traj_length = run_ais_chain(
        model,
        loader,
        mode='forward',
        schedule=forward_schedule,
        hparams=hparams,
        task_params=task_params,
        start_state=start_state,
        init_weights=init_weights,
        init_step_size=init_step_size,
        init_history=init_history,
        init_traj_length=init_traj_length,
        step_sizes_dir=step_sizes_path,
        prior_dist=prior_dist,
        rep_model=rep_model)

    lower_bounds = list()

    for forward in forward_logws:
        lower_bounds.append(forward.mean().cpu().numpy())

    lower_bounds_avg = np.mean(lower_bounds)
    note_taking('run_ais_forward: Average lower bound on {} data: {}'.format(
        data, lower_bounds_avg))

    forward_logws_array = [tensor.cpu().numpy() for tensor in forward_logws]
    forward_logws_array = np.stack(forward_logws_array, axis=0)
    npy_dir = hparams.messenger.arxiv_dir + "forward_ais/"
    init_dir(npy_dir)
    npy_path = npy_dir + ("beta_" + str(hparams.messenger.beta)
                          if hparams.messenger.beta is not None else
                          "") + "_on_" + str(data) + ".npz"
    np.savez(npy_path, forward_logws_array)
    return lower_bounds_avg, approx_post_zs, simulated_data, AIS_weights, epsilon, accept_hist, current_traj_length
