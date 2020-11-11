# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import torch
from ..utils.vae_utils import log_normal_likelihood
from ..utils.computation_utils import normalize_logws, singleton_repeat, normalized_weights_test
from ..utils.experiment_utils import note_taking, plot_rate_distrotion, AISTracker, plot_both
from .bdmc import *
from .ais import run_ais_forward
from ..algorithms.mixture_prior_rd import get_mix_prior
from ..algorithms.analytical_linear_vae import run_analytical_rd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ais_rate_distortion(normalizing_constant_dict, model, beta, approx_post_zs,
                        data_loader, AIS_weights, hparams, data, rep_model,
                        chains_tracker):
    """ 
    Compute rate and distortion based on AIS posterior and normalizing constants. 
    """
    z_zeros = torch.zeros(approx_post_zs[0].size())
    rd_params = hparams.rd
    KLD_list = list()
    distortion_list = list()

    for i, (batch, _) in enumerate(data_loader, 0):
        approx_zs = approx_post_zs[i]
        log_p_zs = log_normal_likelihood(approx_zs, z_zeros, z_zeros)
        x_mean, x_logvar = model.decode(approx_zs)
        normalized_AIS_weights, normalizing_constants_logws = normalize_logws(
            AIS_weights[i], rd_params.batch_size, rd_params.n_chains, hparams)
        if hparams.test_logw:
            note_taking("Testing logw for beta={}".format(beta))
            normalized_weights_test(normalized_AIS_weights, rd_params.n_chains,
                                    rd_params.batch_size, None)
        flattened_normalized_AIS_weights = torch.flatten(normalized_AIS_weights)
        if hparams.test_logw:
            note_taking("Testing flattened logw for beta={}".format(beta))
            normalized_weights_test(
                normalized_AIS_weights.view(rd_params.n_chains,
                                            rd_params.batch_size, -1),
                rd_params.n_chains, rd_params.batch_size, None)

        distortion_expectation = ais_distortion(
            flattened_normalized_AIS_weights, batch, x_mean, x_logvar,
            hparams.rd.target_dist, rd_params.batch_size, rep_model,
            chains_tracker)
        distortion_list.append(distortion_expectation)

        if hparams.simplified_rate:
            KLD_weighted = ais_rate_simplified(normalizing_constants_logws,
                                               distortion_expectation, beta)

        else:
            if hparams.rd.target_dist == "joint_xz":
                KLD_weighted = ais_rate_NLL(
                    normalizing_constants_logws, model, batch, x_mean, x_logvar,
                    beta, approx_zs, flattened_normalized_AIS_weights, log_p_zs,
                    hparams, data)
            else:
                KLD_weighted = ais_rate_MSE(
                    normalizing_constants_logws, model, batch, x_mean, x_logvar,
                    beta, approx_zs, flattened_normalized_AIS_weights, log_p_zs,
                    hparams, data)

        KLD_list.append(KLD_weighted)

        if hparams.every_chain:
            chains_tracker.track_logw(AIS_weights[i])
            note_taking(
                "AIS_weights[i] {} at beta {}".format(AIS_weights[i], beta),
                print_=False)
            chains_tracker.track_logZ(normalizing_constants_logws)

    cated_KLD_expecation = torch.cat(KLD_list, dim=0)
    cated_distortion = torch.cat(distortion_list, dim=0)
    KLD_expecation = torch.mean(
        cated_KLD_expecation, dim=0).cpu().numpy().item()
    distortion_expectation = torch.mean(
        cated_distortion, dim=0).cpu().numpy().item()

    return KLD_expecation, distortion_expectation


def ais_rate_NLL(normalizing_constants_logws, model, batch, x_mean, x_logvar,
                 beta, approx_zs, flattened_normalized_AIS_weights, log_p_zs,
                 hparams, data):
    """ 
    Compute rate for NLL distortion metric. 
     """
    x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
    log_likelihood = log_normal_likelihood(batch, x_mean, x_logvar_full)
    log_likelihood = log_likelihood * beta
    normalizing_constants_logws = torch.squeeze(normalizing_constants_logws)
    normalizing_constants_logws = torch.flatten(
        singleton_repeat(normalizing_constants_logws, hparams.rd.n_chains))
    log_q_zs_lower = log_p_zs + log_likelihood - normalizing_constants_logws
    KLD_weighted = torch.sum(
        flattened_normalized_AIS_weights * (log_q_zs_lower - log_p_zs),
        dim=0,
        keepdim=True) / hparams.rd.batch_size

    return KLD_weighted


def ais_rate_MSE(normalizing_constants_logws, model, batch, x_mean, x_logvar,
                 beta, approx_zs, flattened_normalized_AIS_weights, log_p_zs,
                 hparams, data):
    """ 
    Compute rate for MSE distortion metric. 
     """
    MSE = torch.sum((batch - x_mean)**2, dim=1)
    normalizing_constants_logws = torch.squeeze(normalizing_constants_logws)
    normalizing_constants_logws = torch.flatten(
        singleton_repeat(normalizing_constants_logws, hparams.rd.n_chains))
    log_q_zs = log_p_zs + -MSE * beta - normalizing_constants_logws
    KLD_weighted = torch.sum(
        flattened_normalized_AIS_weights * (log_q_zs - log_p_zs),
        dim=0,
        keepdim=True) / hparams.rd.batch_size

    return KLD_weighted


def ais_rate_simplified(normalizing_constants_logws, distortion, beta):
    """ 
    Compute rate for MSE distortion metric. 
     """
    return -torch.mean(normalizing_constants_logws) - distortion * beta


def ais_distortion(flattened_normalized_AIS_weights,
                   batch,
                   x_mean,
                   x_logvar,
                   distortion_func,
                   batch_size,
                   rep_model=None,
                   tracker=None):
    if distortion_func == "joint_xz" or distortion_func == "mix_prior":
        x_logvar_full = torch.zeros(x_mean.size()) + x_logvar
        distortion = -log_normal_likelihood(batch, x_mean, x_logvar_full)

    else:
        if "fid" in distortion_func:
            gen_rep = rep_model.get_rep_2(x_mean)
            data_rep = rep_model.get_rep_2(batch)
            distortion = torch.sum((gen_rep - data_rep)**2, dim=1)
        else:
            distortion = torch.sum((batch - x_mean)**2, dim=1)

    distortion_weighted = distortion * flattened_normalized_AIS_weights
    distortion_expectation = torch.sum(distortion_weighted, dim=0, keepdim=True)

    if tracker is not None:
        tracker.track_distortion(distortion)

    return distortion_expectation / batch_size


def run_ais_rd_oneshot(model,
                       hparams,
                       data,
                       rd_data_loader,
                       tempreture_dict,
                       writer,
                       rd_ais_results,
                       rep_model=None):
    """ Compute the entire rate distortion curve in a single AIS run
    """

    prior_dist = None
    if hparams.mixture_weight is not None:
        prior_dist = get_mix_prior(hparams, model)
    chains_tracker = None
    if hparams.every_chain:
        chains_tracker = AISTracker(hparams)
    for i in range(hparams.rd.num_betas):
        beta = hparams.messenger.beta_list[i]
        if hparams.verbose:
            note_taking("Running rd-ais with beta value {} on {} data".format(
                beta, data))
        hparams.messenger.beta = beta
        """
        About to read data.
        hparams.messenger.fix_data_simulation is for reproducibility:
        making sure we are always using exactly the same data across betas 
        and between ais and analytical. 
        This is an extra step taken for reproducibility. 
        """

        if i == 0:
            hparams.messenger.fix_data_simulation = False
            lower, approx_post_zs, data_used, AIS_weights, epsilon, accept_hist, traj_length = run_ais_forward(
                model,
                rd_data_loader,
                hparams,
                forward_schedule=tempreture_dict[beta],
                data="rd_" + data,
                task_params=hparams.rd,
                prior_dist=prior_dist,
                rep_model=rep_model)
            hparams.messenger.fix_data_simulation = True

        else:

            lower, approx_post_zs, _, AIS_weights, epsilon, accept_hist, traj_length = run_ais_forward(
                model,
                data_used,
                hparams,
                forward_schedule=tempreture_dict[beta],
                data="rd_" + data,
                task_params=hparams.rd,
                start_state=approx_post_zs,
                init_weights=AIS_weights,
                init_step_size=epsilon,
                init_history=accept_hist,
                init_traj_length=traj_length,
                prior_dist=prior_dist,
                rep_model=rep_model)

        with torch.no_grad():
            normalizing_constant_dict = {"lower": lower}

            if beta == 1:
                note_taking(
                    "normalizing constant for {} data at beta=1: lower:{}".
                    format(data, lower))

            rd_ais_results["rd_const_dict"]["lower"].append(lower)
            rate, distortion = ais_rate_distortion(
                normalizing_constant_dict, model, beta, approx_post_zs,
                data_used, AIS_weights, hparams, data, rep_model,
                chains_tracker)

            if beta == 1:
                note_taking(
                    "For {} data at beta=1: Rate:{}, D({}):{}, -(R+D)={}".
                    format(data, rate, hparams.rd.target_dist, distortion,
                           -rate - distortion))

            rd_ais_results["rate_list"].append(rate)
            writer.add_scalar('rd_on_{}/ais_lower'.format(data), rate,
                              distortion)

            rd_ais_results["distortion_list"].append(distortion)
            hparams.messenger.rd_betas_done.append(beta)
            current_betas = hparams.messenger.rd_betas_done
            note_taking("Finished with beta={}".format(beta))

            if (len(current_betas)) > 1:
                if hparams.rd.target_dist == "joint_xz":
                    dist_func = "neglkh"
                else:
                    dist_func = hparams.rd.target_dist
                plot_rate_distrotion(
                    hparams,
                    current_betas,
                    rd_ais_results["rate_list"],
                    rd_ais_results["distortion_list"],
                    data,
                    metric=dist_func)

            if hparams.verbose:
                note_taking(
                    "MSE rd-ais with beta={} on {} data done. rate={}, distortion={}"
                    .format(beta, data, rate, distortion))

    if hparams.analytic_rd_curve:
        analytic_rate_list, analytic_distortion_list = run_analytical_rd(
            model, hparams, data, rd_data_loader, writer)
        plot_both(hparams, hparams.messenger.beta_list, analytic_rate_list,
                  analytic_distortion_list, rd_ais_results["rate_list"],
                  rd_ais_results["distortion_list"], data)

    if hparams.every_chain:
        if hparams.analytic_rd_curve:
            chains_tracker.plot_with_analytical(analytic_rate_list,
                                                analytic_distortion_list)
        else:
            chains_tracker.plot_rd_chains()

    summery_npy_path = hparams.messenger.arxiv_dir + (
        (data + "_") if data is not None else '') + "ais_rd_summery.npz"
    np.savez(summery_npy_path, rd_ais_results["rd_const_dict"]["lower"],
             rd_ais_results["rate_list"], rd_ais_results["distortion_list"],
             hparams.messenger.beta_list)
    temp_result_dict = {
        "rd_const_dict": rd_ais_results["rd_const_dict"],
        "rate_list": rd_ais_results["rate_list"],
        "distortion_list": rd_ais_results["distortion_list"],
        "beta_list": hparams.messenger.beta_list
    }
    hparams.messenger.result_dict.update({"rd_ais_" + data: temp_result_dict})
    try:
        del data_used
    except:
        pass

    return rd_ais_results["rate_list"], rd_ais_results[
        "distortion_list"], rd_ais_results["rd_const_dict"]["lower"]
