# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from .plots.registry import get_plots
from .hparams.registry import get_hparams
from .hparams.hparam import Hparam as hp
from datetime import datetime
import subprocess
import time
from .utils.experiment_utils import note_taking, init_dir, print_hparams, logging
import argparse
import sys
import logging as python_log

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--hparam_set", default="default", type=str)
parser.add_argument("--e_name")
args = parser.parse_args()
plot_params = get_plots(args.hparam_set)
plot_params.hparam_set = args.hparam_set


def initialze_plots(plot_params, args):
    """ 
    Naming convention: everything ending with "dir" ends with "/"
    """
    plot_dir = plot_params.output_root_dir + "/plots/"
    if plot_params.group_list:
        for subgroup in plot_params.group_list:
            plot_dir += (subgroup + "/")

    plot_dir = plot_dir + args.hparam_set + (
        "_" + str(args.e_name) if args.e_name is not None else "") + "/"
    init_dir(plot_dir)

    plot_params.extra_group = plot_params.group_list
    if plot_params.extra_group is not None:
        extra_dir = plot_params.output_root_dir + "/plots/"
        for subgroup in plot_params.extra_group:
            extra_dir += (subgroup + "/")
        init_dir(extra_dir)
    plot_params.extra_dir = extra_dir
    log_path = plot_dir + "plot_log.txt"
    python_log.basicConfig(
        filename=log_path,
        filemode='a',
        level=python_log.INFO,
        format='%(message)s')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        git_label = subprocess.check_output(
            ["cd " + dir_path + " && git describe --always && cd .."],
            shell=True).strip()
        if plot_params.verbose:
            note_taking("The git label is {}".format(git_label))
    except:
        note_taking("WARNING! Encountered unknwon error recording git label...")

    if plot_params.verbose:
        print_hparams(plot_params.to_dict(), None)
    plot_params.plot_dir = plot_dir
    plot_params.dir_path = dir_path
    plot_params.git_label = git_label
    logging(
        dict({
            "hparam_set": args.hparam_set
        }), plot_params.to_dict(), plot_dir, dir_path)


def find_path(hparams,
              hparam_set,
              data="test",
              metric="neglkh",
              BDMC_exception=None):

    results_dir = hparams.output_root_dir + "/results_out/"
    if hparams.group_list:
        for subgroup in hparams.group_list:
            results_dir += (subgroup + "/")

    if BDMC_exception is not None:
        BDMC_dir = results_dir + BDMC_exception
    else:
        BDMC_dir = results_dir + "BDMC_" + hparam_set
    results_dir = results_dir + hparam_set + "/"

    arxiv_dir = results_dir + "result_arxiv/"
    npz_path = arxiv_dir + hparam_set + '_rd_ais_' + metric + (
        ('_on_' + data) if data is not None else '') + '.npz'

    return npz_path, BDMC_dir


def generate_paths(plot_params, metric=None):
    """ 
    When metric is None, the original metric used in the 
    experiment will be loaded. 
     """
    paths_list = list()
    BDMC_dir_list = list()
    load_metric = True if metric is None else False
    for i, hparam_set in enumerate(plot_params.exp_list, 0):
        hparams = get_hparams(hparam_set)
        if load_metric:
            metric = hparams.rd.target_dist
        if plot_params.BDMC_exceptions is not None:
            BDMC_exception = plot_params.BDMC_exceptions[i]
        else:
            BDMC_exception = None
        path, BDMC_path = find_path(hparams, hparam_set, plot_params.data,
                                    metric, BDMC_exception)
        paths_list.append(path)
        BDMC_dir_list.append(BDMC_path)
    return paths_list, BDMC_dir_list, metric


def build_colormap(start_color, end_color, num_colors):
    """ 
    return a list of RGB colors
    """
    color_array = np.zeros((3, num_colors))
    for i in range(3):
        color_array[i] = np.linspace(
            start_color[i], end_color[i], num=num_colors)
    color_list = [tuple(color_array[:, i]) for i in range(num_colors)]

    return color_list


def sparcify(percentage, D_list, R_list, betas):
    max_D = np.amax(D_list)
    min_D = np.amin(D_list)
    num_points = len(D_list)
    target_num = num_points * percentage
    target_unif = np.linspace(min_D, max_D, target_num)
    D_array = np.transpose(
        np.repeat(np.expand_dims(D_list, axis=0), target_num, axis=0))
    idx_array = np.argmin((np.abs(D_array - target_unif)), axis=0)
    sparse_R = R_list[idx_array]
    sparse_D = D_list[idx_array]
    return np.flip(sparse_D), np.flip(sparse_R), np.flip(betas[idx_array])


def get_beta_index(BDMC_dir, beta_list, num_BDMC_points, cut=False):

    bdmc_hparams = hp()
    bdmc_hparams.restore(BDMC_dir + "/init_hparams.json")
    if bdmc_hparams.beta_indexes is None:
        select_indexes = np.linspace(
            0, len(beta_list) - 1, num_BDMC_points, dtype=int)
        note_taking("Will annotate BDMC for betas from the original index: {}".
                    format(select_indexes))
    else:
        select_indexes = bdmc_hparams.beta_indexes
        if cut:
            note_taking(
                "Adjusting for the distortion/rate limit cut, the original indexes are: {}"
                .format(select_indexes))
            original_num_betas = bdmc_hparams.rd.num_betas
            cut_num_betas = len(beta_list)
            adjusted_indexes = np.array(
                select_indexes) + cut_num_betas - original_num_betas
            for (i, index) in enumerate(adjusted_indexes):
                if index < 0:
                    adjusted_indexes[i] = 0
            select_indexes = np.floor(adjusted_indexes).astype(int).tolist()
            note_taking("Adjusted indexes: {}".format(select_indexes))
    # compensate if there is distortion limit
    note_taking("Will annotate BDMC for betas: {}".format(
        beta_list[select_indexes]))
    return select_indexes


def generate_a_plot(plot_params, metric=None):

    target_exp_paths, BDMC_dir_list, metric = generate_paths(
        plot_params, metric)
    plt.figure()

    for (i, path) in enumerate(target_exp_paths):
        note_taking("loading from: {}".format(path))

        try:
            rd = np.load(path)
        except:
            note_taking(
                "WARNING!! \n Error loading rd data from {} \n".format(path))
        D_list = rd['arr_2']
        try:
            R_list = rd['arr_1'].item().get('lower')
        except:
            R_list = rd['arr_1']
        betas = rd['arr_0']

        # Only specify one of the below two.
        if plot_params.distortion_limit is not None:
            idx = (np.abs(D_list - plot_params.distortion_limit)).argmin()
            note_taking("Distortion limit={}, at beta={} at index {}".format(
                plot_params.distortion_limit, betas[idx], idx))
            D_list = D_list[idx:]
            R_list = R_list[idx:]
            betas = betas[idx:]

        if plot_params.rate_limit is not None:
            idx = (np.abs(R_list - plot_params.rate_limit)).argmin()
            note_taking("Rate limit={}, at beta={} at index {}".format(
                plot_params.rate_limit, betas[idx], idx))
            D_list = D_list[idx:]
            R_list = R_list[idx:]
            betas = betas[idx:]

        color = plot_params.colors[i % len(plot_params.colors)]
        linestyle = plot_params.linestyle[i % len(plot_params.linestyle)]

        annotate_BDMC = plot_params.annotate_BDMC
        if annotate_BDMC:
            BDMC_path = BDMC_dir_list[i] + "/result_arxiv/BDMC.npz"

            try:
                note_taking("Loading BDMC from: {}".format(BDMC_path))
                BDMC_dict = np.load(BDMC_path)

            except:
                note_taking("No BDMC data found. ")
                annotate_BDMC = False
        if annotate_BDMC:
            gap_array = BDMC_dict["gap_np"]
            if plot_params.distortion_limit is not None:
                beta_indexes = get_beta_index(BDMC_dir_list[i], betas,
                                              len(gap_array), idx)
            else:
                beta_indexes = get_beta_index(BDMC_dir_list[i], betas,
                                              len(gap_array))
            for i, gap in enumerate(gap_array, 0):
                plt.annotate(
                    str(format(gap, plot_params.annotate_decimal)),
                    xy=(D_list[beta_indexes[i]], R_list[beta_indexes[i]]))
                plt.plot(
                    D_list[beta_indexes[i]],
                    R_list[beta_indexes[i]],
                    color=color,
                    marker="*",
                    markersize=plot_params.markersize)
        if plot_params.sparse is not None:
            D_list, R_list, betas = sparcify(plot_params.sparse, D_list, R_list,
                                             betas)
            note_taking("Sparsified data. Now only have {} points".format(
                len(D_list)))
        plt.plot(
            D_list,
            R_list,
            color=color,
            linestyle=linestyle,
            label=plot_params.labels[i],
            linewidth=plot_params.linewidth)
    leg = plt.legend(
        loc=plot_params.legend_loc,
        ncol=plot_params.legend_ncol,
        mode=plot_params.legend_mode,
        shadow=True,
        fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if plot_params.title is not None:
        plt.title(plot_params.title, fontsize=plot_params.title_font)
    plt.xlabel(plot_params.xaxis, fontsize=plot_params.axis_font)
    plt.ylabel(plot_params.yaxis, fontsize=plot_params.axis_font)
    plt.yscale(plot_params.yscale)
    if plot_params.y_limit is not None:
        plt.ylim([plot_params.y_limit[0], plot_params.y_limit[1]])
    if plot_params.x_limit is not None:
        plt.xlim([plot_params.x_limit[0], plot_params.x_limit[1]])
    plt.grid(b=True, axis="both")
    plt.tight_layout()
    save_name = args.hparam_set + "_" + metric
    plt.savefig(plot_params.plot_dir + save_name + '.pdf', bbox_inches='tight')
    if plot_params.extra_group is not None:
        plt.savefig(
            plot_params.extra_dir + save_name + '.pdf', bbox_inches='tight')
    plt.close()
    note_taking("{} plot saved at: {}".format(plot_params.title,
                                              plot_params.plot_dir))


if __name__ == '__main__':

    initialze_plots(plot_params, args)

    if plot_params.start_color is not None:
        plot_params.colors = build_colormap(plot_params.start_color,
                                            plot_params.end_color,
                                            len(plot_params.labels))
        note_taking(
            "OVERWRITING plot_params.colors to colors from the colormap: {}".
            format(plot_params.colors))

    generate_a_plot(plot_params)

    logging(
        dict({
            "hparam_set": args.hparam_set
        }),
        plot_params.to_dict(),
        plot_params.plot_dir,
        plot_params.dir_path,
        stage="final")
