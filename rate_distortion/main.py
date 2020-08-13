# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from datetime import datetime
import subprocess
import time
import torch
from torch import optim
from .hparams.registry import get_hparams
from .data.load_data import *
from .utils.anneal_schedules import get_schedule
from .utils.experiment_utils import *
from .algorithms.rate_distortion import run_ais_rd_oneshot
from .algorithms.analytical_linear_vae import run_analytical_rd
from .algorithms.train_vae import train_and_test
from .algorithms.bdmc import run_BDMC_betas
from .algorithms.train_mnist_CNN import get_mnist_representation
from .algorithms.rd_baseline import run_rd_baseline
import argparse
import random
import sys
from .utils.gan_utils import gan_bridge
from .utils.aae_utils import aae_bridge
from .utils.vae_utils import prepare_vae

sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--hparam_set", default="default", type=str)
parser.add_argument("--e_name")
args = parser.parse_args()
args_dict = vars(args)
hparams = get_hparams(args.hparam_set)
set_random_seed(hparams.random_seed)


def main(writer, hparams):

    cnn_model = None
    if hparams.cnn_model_name is not None:
        if "mnist" in hparams.cnn_model_name:
            cnn_model = get_mnist_representation(hparams.representation,
                                                 hparams)

    # Make sure to include vae in your VAE models' names, same for gan and aae.
    if hparams.original_experiment:
        if "vae" in hparams.model_name:
            note_taking("About to run experiment on {} with z size={}".format(
                hparams.model_name, hparams.model_train.z_size))
            model = prepare_vae(writer, hparams)

        elif "gan" in hparams.model_name:
            model = gan_bridge(hparams)
            note_taking(
                "About to run GAN experiment on {} with z size={}".format(
                    hparams.model_name, hparams.model_train.z_size))
        elif "aae" in hparams.model_name:
            model = aae_bridge(hparams)
            note_taking(
                "About to run AAE experiment on {} with z size={}".format(
                    hparams.model_name, hparams.model_train.z_size))
    else:
        model = load_user_model(hparams)
        note_taking("Loaded user model {} with z size={}".format(
            hparams.model_name, hparams.model_train.z_size))
    sample_images(hparams, model, hparams.epoch, best=False)
    x_var = torch.exp(model.x_logvar).detach().cpu().numpy().item()
    note_taking(
        "Loaded the generative model: {} with decoder variance {}".format(
            hparams.model_name, x_var))
    model.eval()

    if hparams.rd.rd_data_list:
        hparams.messenger.running_rd = True

        if hparams.tempreture_path is None:
            t_schedule = get_schedule(hparams.rd)
            tempreture_dict = tempreture_scheduler_beta(hparams.rd.max_beta,
                                                        t_schedule, hparams)

            tempreture_npy_path = hparams.messenger.arxiv_dir + "tempreture.npz"
            np.savez(tempreture_npy_path, tempreture_dict)
            hparams.messenger.tempreture_npy_path = tempreture_npy_path
            note_taking(
                "tempreture schedule saved at: {}".format(tempreture_npy_path))
        else:
            tempreture_dict = np.load(hparams.tempreture_path)["arr_0"]
            note_taking("tempreture schedule loaded from: {}".format(
                hparams.tempreture_path))
            for index, v in np.ndenumerate(tempreture_dict):
                temp = v
            tempreture_dict = temp
        betas = hparams.messenger.beta_list
        if hparams.rd_baseline:
            data_loader = load_rd_data(model, hparams, "test")

            lambdas_list = infer_lambda_range(1. / hparams.rd.max_beta,
                                              1. / hparams.rd.min_beta,
                                              hparams.rd.num_betas, 3)
            hparams.messenger.beta_list = 1. / np.flip(lambdas_list, axis=0)

            note_taking(
                "About to run the baseline, overwriting the beta list: {}".
                format(hparams.messenger.beta_list))
            base_rates, baseline_distortions = run_rd_baseline(
                model, data_loader, hparams)
            if hparams.analytic_rd_curve:
                rd_ais_results = init_rd_ais_results("test")
                ais_rate_list, ais_distortion_list, lower_NC = run_ais_rd_oneshot(
                    model,
                    hparams,
                    "test",
                    data_loader,
                    tempreture_dict,
                    writer,
                    rd_ais_results,
                    rep_model=cnn_model)

                analytic_rate_list, analytic_distortion_list = run_analytical_rd(
                    model, hparams, "test", data_loader, writer)
                plot_both_baseline(hparams, analytic_rate_list,
                                   analytic_distortion_list, ais_rate_list,
                                   ais_distortion_list, base_rates,
                                   baseline_distortions, "test")
        else:
            for data in hparams.rd.rd_data_list:
                rd_ais_results = init_rd_ais_results(data)
                if data != "simulate":
                    note_taking(
                        "About to start running for {} betas, ranging from: [{},{}] on {} data"
                        .format(len(betas), betas[0], betas[-1], data))
                    rd_data_loader = load_rd_data(model, hparams, data)
                    if hparams.compute_data_stats:
                        mean, std, data_max, data_min = compute_data_stats(
                            rd_data_loader, data)
                        note_taking(
                            "verifying {} data stats... mean: {} \n std: {} \n max: {} \n min: {} \n"
                            .format(data, mean, std, data_max, data_min))
                    ais_rate_list, ais_distortion_list, lower_NC = run_ais_rd_oneshot(
                        model,
                        hparams,
                        data,
                        rd_data_loader,
                        tempreture_dict,
                        writer,
                        rd_ais_results,
                        rep_model=cnn_model)
                else:
                    upper_NC, lower_NC, BDMC_betas, select_indexs = run_BDMC_betas(
                        model, tempreture_dict, hparams, cnn_model)
                hparams.messenger.rd_betas_done = list()
            end_time = datetime.now()
            hour_summery = compute_duration(hparams.messenger.start_time,
                                            end_time)
            note_taking("RD on {} data finished. Took {} hours".format(
                data, hour_summery))
    logging(
        args_dict,
        hparams.to_dict(),
        hparams.messenger.results_dir,
        hparams.messenger.dir_path,
        stage="final")


if __name__ == '__main__':

    writer = initialze_run(hparams, args)
    start_time = datetime.now()
    hparams.messenger.start_time = start_time
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        git_label = subprocess.check_output(
            ["cd " + dir_path + " && git describe --always && cd .."],
            shell=True).strip()
        if hparams.verbose:
            note_taking("The git label is {}".format(git_label))
    except:
        note_taking("WARNING! Encountered unknwon error recording git label...")

    main(writer, hparams)
    end_time = datetime.now()
    hour_summery = compute_duration(start_time, end_time)
    writer_json_path = hparams.messenger.tboard_path + "/tboard_summery.json"
    writer.export_scalars_to_json(writer_json_path)
    writer.close()
    note_taking(
        "Experiment finished, results written at: {}. Took {} hours".format(
            hparams.messenger.results_dir, hour_summery))
