# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import register
from .defaults import *
from .hparam import Hparam as hp

# ----------------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------------


def default_baseline():
    Hparam = default_experiment()
    Hparam.baseline_samples = 100
    Hparam.rd_baseline = True
    Hparam.model_train.epochs = 50
    Hparam.group_list = ["icml", "baselines"]
    Hparam.learning_rate = 1e-4
    Hparam.analytical_rate = True
    Hparam.rd.target_dist = "baseline"
    Hparam.baseline_reuse = True
    Hparam.rd.max_beta = 50000
    Hparam.rd.min_beta = 0.001
    Hparam.rd.num_betas = 25
    Hparam.original_experiment = True
    return Hparam


def vae_experiment():
    Hparam = default_baseline()
    Hparam.model_name = "deep_vae"
    Hparam.chkt_step = -2  #-2 means take the best, -1 means take the latest
    Hparam.train_print_freq = 100
    Hparam.start_checkpointing = 0
    Hparam.checkpointing_freq = 100
    Hparam.resume_training = True
    Hparam.train_first = True
    Hparam.n_test_batch = 10
    return Hparam


def VAE_RD():
    Hparam = vae_experiment()
    Hparam.model_name = "deep_vae"
    return Hparam


def gan_experiment():
    Hparam = default_baseline()
    Hparam.model_name = "gan"
    return Hparam


def GAN_MNIST():
    Hparam = gan_experiment()
    return Hparam


@register("deep_gan2_GP_baseline_icml")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    return Hparam


@register("deep_gan10_GP_baseline_icml")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    return Hparam


@register("deep_gan100_GP_baseline_icml")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_debug4_mnist_gp_LT_delta-0x450c084a_base-0x461907a5_LT/LT_delta-0x450c084a_base-0x461907a5_LT_netG_epoch_106.pth"
    return Hparam


@register("vae2_rd_baseline_icml")
def vae2_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    return Hparam


@register("vae10_rd_baseline_icml")
def vae10_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    return Hparam


@register("vae100_rd_baseline_icml")
def la12_MSE_vae100_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 100
    Hparam.load_hparam_name = "icml_vae100"
    return Hparam
