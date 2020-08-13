# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/hmc.py
# and the R code in R. Neil's paper https://arxiv.org/pdf/1206.1901.pdf

import torch
import numpy as np
from ..utils.experiment_utils import note_taking


def hmc_trajectory(current_z, batch, t, current_v, anneal_ops, epsilon, hparams,
                   task_params):
    """compute HMC trajectory

    Args:
        batch: a batch of data
        t: tempreture
        anneal_ops: annealing operations
        epsilon: (adaptive) step size
    """

    eps = epsilon.view(-1, 1)
    z = current_z
    grad_U_temp = anneal_ops.grad_U(z, batch, t)
    v = current_v - 0.5 * grad_U_temp.mul(eps)
    for i in range(1, task_params.leap_steps + 1):
        z = z + v.mul(eps)
        grad_U_temp = anneal_ops.grad_U(z, batch, t)

        if i < task_params.leap_steps:
            v = v - grad_U_temp.mul(eps)

    v = v - 0.5 * grad_U_temp.mul(eps)
    v = -v
    z.detach_()
    v.detach_()
    return z, v


def accept_reject(current_z, batch, t, current_v, hmc_z, hmc_v, epsilon,
                  accept_hist, hist_len, anneal_ops, hparams, task_params):
    """Accept/reject based on Hamiltonians for current and propose.

    Args:
        current_z: position before leap frog steps
        batch: a batch of data
        t: tempreture
        current_v: speed before leap frog steps
        hmc_z: position after leap frog steps
        hmc_v: speed AFTER leap frog steps
        epsilon: step size of leap-frog.
        accept_hist: the sum of accept/reject indicators
        anneal_ops: annealing operations 
    """
    # extra precaution
    with torch.no_grad():
        U_1 = anneal_ops.U(current_z, batch, t)
        U_2 = anneal_ops.U(hmc_z, batch, t)
        K_1 = anneal_ops.normalized_kinetic(current_v)
        K_2 = anneal_ops.normalized_kinetic(hmc_v)
        current_Hamil = K_1 + U_1
        propose_Hamil = K_2 + U_2
        prob = torch.exp(current_Hamil - propose_Hamil)
        uniform_sample = torch.rand(prob.size())
        accept = (prob > uniform_sample).type(hparams.dtype)
        next_z = hmc_z.mul(accept.view(
            -1, 1)) + current_z.mul(1. - accept.view(-1, 1))
        accept_hist = accept_hist.add(accept)
        if hparams.monitor_hmc == True:
            avg_accept = accept_hist / hist_len
            avg_avg_accept = torch.mean(avg_accept)
            if avg_avg_accept < 0.60:
                note_taking(
                    "WARNING! HMC unstable at t={},at step={}, average avg_accept={}"
                    .format(t, hist_len, avg_avg_accept))
        if hparams.step_sizes_target is None:
            criteria = (accept_hist / hist_len >
                        task_params.acceptance_prob).type(hparams.dtype)
            adapt = 1.02 * criteria + 0.98 * (1. - criteria)
            epsilon = epsilon.mul(adapt).clamp(1e-4, .5)

        return next_z, epsilon, accept_hist
