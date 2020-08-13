# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from ..hparams.hparam import Hparam as hp
from .registry import register
import numpy as np


def default_style():

    return hp(
        verbose=True,
        output_root_dir="/home/rd_workspace",
        data="test",
        legend_loc='upper right',
        legend_mode=None,
        legend_ncol=1,
        yscale="log",
        annotate_decimal='.3f')


def default_rd_plot():
    plot_params = default_style()
    plot_params.colors = [
        'cyan',
        'dodgerblue',
        'blue',
        'red',
        'magenta',
        'darkviolet',
        'lawngreen',
        'olive',
        'green',
        'y',
        'lightslategray',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'coral',
        'lavender',
        'turquoise',
        'tan',
        'salmon',
        'gold',
    ]

    plot_params.labels = ["plot1", "plot2"]
    plot_params.exp_list = ["exp1", "exp2"]
    plot_params.xaxis = "Distortion (MSE)"
    plot_params.yaxis = "Rate"
    plot_params.group_list = ["icml_plots"]
    plot_params.legend_loc = 'upper right'
    plot_params.linewidth = 1
    plot_params.markersize = 6
    plot_params.axis_font = 12
    plot_params.title_font = 16
    plot_params.linestyle = ['-' for _ in plot_params.exp_list]
    return plot_params


# ----------------------------------------------------------------------------
# paper plots
# ----------------------------------------------------------------------------


@register("GANs_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
        'red',
    ]
    plot_params.linestyle = [
        ':',
        '-.',
        '--',
        '-',
        ':',
        '-.',
        '--',
        '-',
    ]

    plot_params.labels = [
        "GP-Deep-GAN2",
        "GP-Deep-GAN5",
        "GP-Deep-GAN10",
        "GP-Deep-GAN100",
        "GP-Shallow-GAN2",
        "GP-Shallow-GAN5",
        "GP-Shallow-GAN10",
        "GP-Shallow-GAN100",
    ]
    plot_params.exp_list = [
        "icml_deep_gan2_GP_rerun",
        "icml_deep_gan5_GP_rerun",
        "icml_deep_gan10_GP_rerun",
        "icml_deep_gan100_GP_rerun",
        "icml_shallow_gan2_GP_rerun",
        "icml_shallow_gan5_GP_rerun",
        "icml_shallow_gan10_GP_rerun",
        "icml_shallow_gan100_GP_rerun",
    ]
    plot_params.distortion_limit = 60
    return plot_params


@register("GAN_BDMC_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.labels = [
        "GP-GAN2",
        "GP-GAN10",
        "GP-GAN100",
    ]
    plot_params.exp_list = [
        "icml_deep_gan2_GP_rerun",
        "icml_deep_gan10_GP_rerun",
        "icml_deep_gan100_GP_rerun",
    ]

    plot_params.colors = [
        'blue',
        'blue',
        'blue',
    ]

    plot_params.linestyle = [
        ':',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 40
    plot_params.annotate_BDMC = True
    plot_params.sparse = 0.1
    return plot_params


@register("VAE_BDMC_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.labels = [
        "VAE2",
        "VAE10",
        "VAE100",
    ]
    plot_params.exp_list = [
        "icml_vae2_rd_rerun",
        "icml_vae10_rd_rerun",
        "icml_vae100_rerun",
    ]

    plot_params.colors = [
        'red',
        'red',
        'red',
    ]

    plot_params.linestyle = [
        ':',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 40
    plot_params.annotate_BDMC = True
    plot_params.sparse = 0.1
    return plot_params


@register("AAE_BDMC_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.labels = ["AAE2", "AAE10", "AAE100"]
    plot_params.exp_list = [
        "icml_aae2_rerun",
        "icml_aae10_rerun",
        "icml_aae100_rerun",
    ]

    plot_params.colors = [
        'green',
        'green',
        'green',
    ]

    plot_params.linestyle = [
        ':',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 40
    plot_params.annotate_BDMC = True
    plot_params.sparse = 0.1
    return plot_params


@register("GAN_VAE_AAE_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.labels = [
        "GP-GAN2", "GP-GAN10", "GP-GAN100", "VAE2", "VAE10", "VAE100", "AAE2",
        "AAE10", "AAE100"
    ]
    plot_params.exp_list = [
        "icml_deep_gan2_GP_rerun",
        "icml_deep_gan10_GP_rerun",
        "icml_deep_gan100_GP_rerun",
        "icml_vae2_rd_rerun",
        "icml_vae10_rd_rerun",
        "icml_vae100_rerun",
        "icml_aae2_rerun",
        "icml_aae10_rerun",
        "icml_aae100_rerun",
    ]

    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
        'green',
        'green',
        'green',
    ]

    plot_params.linestyle = [
        ':',
        '--',
        '-',
        ':',
        '--',
        '-',
        ':',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 40
    plot_params.sparse = 0.1
    return plot_params


@register("blurry_VAE_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.colors = [
        'darkblue',
        'blue',
        'slateblue',
        'darkorchid',
        'darkviolet',
        'darkmagenta',
    ]
    plot_params.labels = [
        "Original VAE10",
        "G-blur radius=0.1",
        "G-blur radius=0.5",
        "G-blur radius=1",
        "G-blur radius=2",
        "G-blur radius=5",
    ]
    plot_params.exp_list = [
        "icml_vae10_rd_rerun",
        "icml_vae10_blur01_rerun",
        "icml_vae10_blur05_rerun",
        "icml_vae10_blur1_rerun",
        "icml_vae10_blur2_rerun",
        "icml_vae10_blur5_rerun",
    ]
    plot_params.end_color = (1, 0, 0)
    plot_params.start_color = (0, 1, 0)
    plot_params.rate_limit = 1.
    return plot_params


@register("mixture_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.colors = [
        "black",
        'darkblue',
        'mediumblue',
        'blue',
        'slateblue',
        'rebeccapurple',
        'darkorchid',
        'darkviolet',
        'darkmagenta',
        'darkred',
    ]
    plot_params.labels = [
        "Prior",
        "10% Bad Prior",
        "20% Bad Prior",
        "50% Bad Prior",
        "80% Bad Prior",
        "90% Bad Prior",
        "99% Bad Prior",
    ]
    plot_params.end_color = (1, 0, 0)
    plot_params.start_color = (0, 1, 0)

    plot_params.exp_list = [
        "icml_vae10_rd_rerun",
        "icml_vae10_mp_010_rerun",
        "icml_vae10_mp_020_rerun",
        "icml_vae10_mp_050_rerun",
        "icml_vae10_mp_080_rerun",
        "icml_vae10_mp_090_rerun",
        "icml_vae10_mp_099_rerun",
    ]
    plot_params.rate_limit = 1.
    return plot_params


@register("CIFAR10_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
        'green',
        'green',
        'green',
        'darkviolet',
        'darkviolet',
        'darkviolet',
    ]
    plot_params.linestyle = [
        '-.',
        '--',
        '-',
        '-.',
        '--',
        '-',
        '-.',
        '--',
        '-',
        '-.',
        '--',
        '-',
    ]
    plot_params.labels = [
        "GP-DCGAN5", "GP-DCGAN10", "GP-DCGAN100", "BRE-DCGAN5", "BRE-DCGAN10",
        "BRE-DCGAN100", "DCGAN5", "DCGAN10", "DCGAN100", "SN-GAN5", "SN-GAN10",
        "SN-GAN100"
    ]
    plot_params.exp_list = [
        "icml_CIFAR10_dcgan5_GP_rerun", "icml_CIFAR10_dcgan10_GP_rerun",
        "icml_CIFAR10_dcgan100_GP_rerun", "icml_CIFAR10_dcgan5_BRE_rerun",
        "icml_CIFAR10_dcgan10_BRE_rerun", "icml_CIFAR10_dcgan100_BRE_rerun",
        "icml_CIFAR10_dcgan5_V_rerun", "icml_CIFAR10_dcgan10_V_rerun",
        "icml_CIFAR10_dcgan100_V_rerun", "icml_CIFAR10_dcgan5_SN_rerun",
        "icml_CIFAR10_dcgan10_SN_rerun", "icml_CIFAR10_dcgan100_SN_rerun"
    ]
    plot_params.distortion_limit = 450
    return plot_params


@register("GANs_fid_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
    ]
    plot_params.linestyle = [
        ':',
        '--',
        '-',
        ':',
        '--',
        '-',
    ]
    plot_params.labels = [
        "GP-Deep-GAN2",
        "GP-Deep-GAN5",
        "GP-Deep-GAN10",
        "GP-Shallow-GAN2",
        "GP-Shallow-GAN5",
        "GP-Shallow-GAN10",
    ]
    plot_params.exp_list = [
        "fid_icml_deep_gan2_GP_rerun",
        "fid_icml_deep_gan5_GP_rerun",
        "fid_icml_deep_gan10_GP_rerun",
        "fid_icml_shallow_gan2_GP_rerun",
        "fid_icml_shallow_gan5_GP_rerun",
        "fid_icml_shallow_gan10_GP_rerun",
    ]
    plot_params.distortion_limit = 4000
    return plot_params


@register("GVA_fid_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.labels = [
        "GP-GAN2",
        "GP-GAN10",
        "VAE2",
        "VAE10",
        "AAE2",
        "AAE10",
    ]
    plot_params.exp_list = [
        "fid_icml_deep_gan2_GP_rerun",
        "fid_icml_deep_gan10_GP_rerun",
        "fid_icml_vae2_rd_rerun",
        "fid_icml_vae10_rd_rerun",
        "fid_icml_aae2_rerun",
        "fid_icml_aae10_rerun",
    ]

    plot_params.colors = [
        'blue',
        'blue',
        'red',
        'red',
        'green',
        'green',
    ]

    plot_params.linestyle = [
        '--',
        '-',
        '--',
        '-',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 3500
    return plot_params


@register("GAN_baseline_icml_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.labels = [
        "GP-GAN2 AIS",
        "GP-GAN10 AIS",
        "GP-GAN100 AIS",
        "GP-GAN2 Variational",
        "GP-GAN10 Variational",
        "GP-GAN100 Variational",
    ]
    plot_params.exp_list = [
        "icml_deep_gan2_GP_rerun",
        "icml_deep_gan10_GP_rerun",
        "icml_deep_gan100_GP_rerun",
        "deep_gan2_GP_baseline_icml_50p",
        "deep_gan10_GP_baseline_icml_50p",
        "deep_gan100_GP_baseline_icml_50p",
    ]
    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
    ]
    plot_params.linestyle = [
        ':',
        '--',
        '-',
        ':',
        '--',
        '-',
    ]
    plot_params.distortion_limit = 50
    return plot_params


@register("VAE_baseline_icml_rerun")
def test():
    plot_params = default_rd_plot()

    plot_params.labels = [
        "VAE2 AIS",
        "VAE10 AIS",
        "VAE100 AIS",
        "VAE2 Variational",
        "VAE10 Variational",
        "VAE100 Variational",
    ]

    plot_params.exp_list = [
        "icml_vae2_rd_rerun",
        "icml_vae10_rd_rerun",
        "icml_vae100_rerun",
        "vae2_rd_baseline_icml_50p",
        "vae10_rd_baseline_icml_50p",
        "vae100_rd_baseline_icml_50p",
    ]

    plot_params.colors = [
        'blue',
        'blue',
        'blue',
        'red',
        'red',
        'red',
    ]

    plot_params.linestyle = [
        ':',
        '--',
        '-',
        ':',
        '--',
        '-',
    ]

    plot_params.distortion_limit = 50
    return plot_params


@register("icml_random_seeds_rerun")
def test():
    plot_params = default_rd_plot()
    plot_params.colors = [
        'cyan',
        'dodgerblue',
        'blue',
        'red',
        'magenta',
        'darkviolet',
        'lawngreen',
        'olive',
        'green',
        'y',
        'lightslategray',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'coral',
        'lavender',
        'turquoise',
        'tan',
        'salmon',
        'gold',
    ]
    plot_params.labels = [
        "preliminary (rs=6)",
        "loaded, rs=11",
        "loaded, rs=12",
        "loaded, rs=13",
        "loaded, rs=14",
        "loaded, rs=15",
        "loaded, rs=16",
    ]
    plot_params.exp_list = [
        "icml_vae10_rd",
        "icml_vae10_rerun_rs11",
        "icml_vae10_rerun_rs12",
        "icml_vae10_rerun_rs13",
        "icml_vae10_rerun_rs14",
        "icml_vae10_rerun_rs15",
        "icml_vae10_rerun_rs16",
    ]
    return plot_params
