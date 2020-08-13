# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2018 Xuechen Li
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/lxuechen/BDMC/blob/master/bdmc.py

from .ais import run_ais_chain
from ..utils.experiment_utils import note_taking, initialze_BDMC, init_dir
import numpy as np
from ..data.load_data import load_simulate_data


def run_bdmc(model,
             hparams,
             data_loader,
             forward_schedule,
             task_params,
             start_state=None,
             init_weights=None,
             init_step_size=None,
             init_history=None):
    """Run BDMC.

    Args:
        model : Any model with a decoder p(x|z)
        data_loader (iterator): iterator to loop over pairs of data; the first
            entry being `x`, the second being `z` sampled from the true
            posterior `p(z|x)`, which is equivalent as first sample from prior and then 
            sample x from the decoder.
        forward_schedule (list or numpy.ndarray): forward temperature schedule;
    Returns:
        please refer to ais.py. The only difference is bdmc is also returning the upper bound, and the average of lower and upper bound. 
    """
    step_sizes_path = hparams.messenger.step_sizes_dir + "sim_forward_"

    forward_logws, approx_post_zs, simulated_data, importance_weights, epsilon, accept_hist, _ = run_ais_chain(
        model,
        data_loader,
        mode='forward',
        schedule=forward_schedule,
        hparams=hparams,
        task_params=task_params,
        start_state=start_state,
        init_weights=init_weights,
        init_step_size=init_step_size,
        init_history=init_history,
        step_sizes_dir=step_sizes_path)

    # run backward chain
    backward_schedule = np.flip(forward_schedule, axis=0)
    step_sizes_path = hparams.messenger.step_sizes_dir + "sim_backward_"
    backward_logws = run_ais_chain(
        model,
        data_loader,
        mode='backward',
        schedule=backward_schedule,
        hparams=hparams,
        task_params=task_params,
        start_state=None,
        init_weights=None,
        step_sizes_dir=step_sizes_path)

    upper_bounds = list()
    lower_bounds = list()

    # average out w.r.t each batch of data
    for _, (forward, backward) in enumerate(zip(forward_logws, backward_logws)):
        lower_bounds.append(forward.mean().cpu().numpy())
        upper_bounds.append(backward.mean().cpu().numpy())

    # grand average
    upper_bounds_avgs = np.mean(upper_bounds)
    lower_bounds_avgs = np.mean(lower_bounds)
    mean_avgs = (upper_bounds_avgs + lower_bounds_avgs) / 2
    note_taking('Average bounds on simulated data: lower {}, upper {}'.format(
        lower_bounds_avgs, upper_bounds_avgs).center(80))

    forward_logws_array = [tensor.cpu().numpy() for tensor in forward_logws]
    forward_logws_array = np.stack(forward_logws_array, axis=0)
    backward_logws_array = [tensor.cpu().numpy() for tensor in backward_logws]
    backward_logws_array = np.stack(backward_logws_array, axis=0)

    npy_path = hparams.messenger.arxiv_dir + (
        ("result_beta_" + str(hparams.messenger.beta))
        if hparams.messenger.beta is not None else "") + ".npz"
    np.savez(npy_path, forward_logws_array, backward_logws_array)
    return upper_bounds_avgs, mean_avgs, lower_bounds_avgs, approx_post_zs, simulated_data, importance_weights, epsilon, accept_hist


def run_BDMC_betas(model, tempreture_dict, hparams, rep_model):
    BDMC_tempreture_dict, BDMC_betas, select_indexs = initialze_BDMC(
        tempreture_dict, hparams)
    upper_bounds_list = list()
    lower_bounds_list = list()
    for beta in BDMC_betas:
        hparams.messenger.beta = beta
        simulated_data = load_simulate_data(
            model, hparams, beta, simulate_dir=hparams.simulate_dir)
        step_sizes_folder = hparams.messenger.step_sizes_dir + "sim_beta_{}/".format(
            beta)
        step_sizes_path = step_sizes_folder + "forward_"
        init_dir(step_sizes_folder)
        forward_logws, _, _, _, _, _, _ = run_ais_chain(
            model,
            simulated_data,
            mode='forward',
            schedule=BDMC_tempreture_dict[beta],
            hparams=hparams,
            task_params=hparams.rd,
            start_state=None,
            init_weights=None,
            step_sizes_dir=step_sizes_path,
            rep_model=rep_model)

        # run backward chain
        step_sizes_path = step_sizes_folder + "backward_"
        backward_schedule = np.flip(BDMC_tempreture_dict[beta], axis=0)
        backward_logws = run_ais_chain(
            model,
            simulated_data,
            mode='backward',
            schedule=backward_schedule,
            hparams=hparams,
            task_params=hparams.rd,
            start_state=None,
            init_weights=None,
            step_sizes_dir=step_sizes_path,
            rep_model=rep_model)

        upper_bounds = list()
        lower_bounds = list()

        # average out w.r.t each batch of data
        for _, (forward, backward) in enumerate(
                zip(forward_logws, backward_logws)):
            lower_bounds.append(forward.mean().cpu().numpy())
            upper_bounds.append(backward.mean().cpu().numpy())

        # grand average
        upper_bounds_avg = np.mean(upper_bounds)
        lower_bounds_avg = np.mean(lower_bounds)
        note_taking("BDMC for beta={} is: lower={}, upper={}. Gap={}".format(
            beta, lower_bounds_avg, upper_bounds_avg,
            upper_bounds_avg - lower_bounds_avg))
        upper_bounds_list.append(upper_bounds_avg)
        lower_bounds_list.append(lower_bounds_avg)
        upper_NC_np = np.asarray(upper_bounds_list)
        lower_NC_np = np.asarray(lower_bounds_list)
        gap_np = upper_NC_np - lower_NC_np
        npy_path = hparams.messenger.arxiv_dir + "BDMC.npz"
        np.savez(
            npy_path,
            upper_NC_np=upper_NC_np,
            lower_NC_np=lower_NC_np,
            gap_np=gap_np)
    note_taking('betas: {}'.format(BDMC_betas).center(80))
    note_taking('lower bounds: {}'.format(lower_bounds_list).center(80))
    note_taking('upper bounds: {}'.format(upper_bounds_list).center(80))
    note_taking('BDMC gap: {}'.format(gap_np).center(80))
    note_taking('selected indexs: {}'.format(select_indexs).center(80))
    note_taking("BDMC done. Results saved at {}".format(npy_path))
    return upper_bounds_list, lower_bounds_list, BDMC_betas, select_indexs
