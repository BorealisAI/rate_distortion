# Copyright (c) 2018-present, Royal Bank of Canada.
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
