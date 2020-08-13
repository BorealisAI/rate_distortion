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
def vae_experiment():
    Hparam = default_experiment()
    Hparam.model_name = "deep_vae"
    Hparam.chkt_step = -2  #-2 means take the best, -1 means take the latest
    Hparam.train_print_freq = 100
    Hparam.start_checkpointing = 0
    Hparam.checkpointing_freq = 100
    Hparam.learning_rate = 1e-4
    Hparam.resume_training = True
    Hparam.n_test_batch = 10
    Hparam.rd.target_dist = "MSE"
    Hparam.original_experiment = True
    return Hparam


def VAE_RD():
    Hparam = vae_experiment()
    Hparam.model_name = "deep_vae"
    Hparam.group_list = ["icml", "VAEs", "RD"]
    return Hparam


def VAE_g_blur():
    Hparam = vae_experiment()
    Hparam.train_first = False
    Hparam.model_name = "blurry_vae"
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam.model_train.z_size = 10
    Hparam.group_list = ["icml", "VAEs", "guassian_blur"]
    Hparam.blur_std = 1
    return Hparam


def VAE_mix_prior():
    Hparam = vae_experiment()
    Hparam.train_first = False
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam.model_train.z_size = 10
    Hparam.rd.target_dist = "mix_prior"
    Hparam.group_list = ["icml", "VAEs", "mix_prior"]
    Hparam.mixture_weight = 0.01  #weight of the bad prior
    Hparam.bad_std = 10
    Hparam.rd.target_dist = "mix_prior_MSE"
    return Hparam


def VAE_RD_rerun():
    Hparam = VAE_RD()
    Hparam.group_list = ["icml", "VAEs", "RD"]
    Hparam.train_first = False
    Hparam.random_seed = 7
    return Hparam


def VAE_g_blur_rerun():
    Hparam = VAE_g_blur()
    Hparam.group_list = ["icml", "VAEs", "guassian_blur"]
    Hparam.random_seed = 7
    return Hparam


def VAE_mix_prior_rerun():
    Hparam = VAE_mix_prior()
    Hparam.group_list = ["icml", "VAEs", "mix_prior"]
    Hparam.random_seed = 7
    return Hparam


# ----------------------------------------------------------------------------
# mix prior
# ----------------------------------------------------------------------------


@register("icml_vae10_mp_010")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.1
    return Hparam


@register("icml_vae10_mp_020")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.20
    return Hparam


@register("icml_vae10_mp_050")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.5
    return Hparam


@register("icml_vae10_mp_080")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.8
    return Hparam


@register("icml_vae10_mp_090")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.90
    return Hparam


@register("icml_vae10_mp_099")
def __doesnotmatter():
    Hparam = VAE_mix_prior()
    Hparam.mixture_weight = 0.99
    return Hparam


# # ----------------------------------------------------------------------------
# # 0630 blurry vae
# # ----------------------------------------------------------------------------


@register("icml_vae10_blur5")
def __dnm():
    Hparam = VAE_g_blur()
    Hparam.blur_std = 5
    return Hparam


@register("icml_vae10_blur2")
def __dnm():
    Hparam = VAE_g_blur()
    Hparam.blur_std = 2
    return Hparam


@register("icml_vae10_blur1")
def __dnm():
    Hparam = VAE_g_blur()
    Hparam.blur_std = 1
    return Hparam


@register("icml_vae10_blur01")
def __dnm():
    Hparam = VAE_g_blur()
    Hparam.blur_std = 0.1
    return Hparam


@register("icml_vae10_blur05")
def __dnm():
    Hparam = VAE_g_blur()
    Hparam.blur_std = 0.5
    return Hparam


# ----------------------------------------------------------------------------
# rd
# ----------------------------------------------------------------------------


@register("icml_vae10_rd")
def vae10_rd():
    Hparam = VAE_RD()
    Hparam.train_first = True
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    return Hparam


@register("icml_vae2_rd")
def vae2_rd():
    Hparam = VAE_RD()
    Hparam.train_first = True
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    return Hparam


@register("icml_vae100")
def __doesnotmatter():
    Hparam = VAE_RD()
    Hparam.train_first = True
    Hparam.model_train.z_size = 100
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


# ----------------------------------------------------------------------------
# linear vae analytical verification
# ----------------------------------------------------------------------------
@register("icml_linear_vae100")
def vae100_linear_analytical():
    Hparam = VAE_RD()
    Hparam.train_first = True
    Hparam.double_precision = True
    Hparam.model_train.z_size = 100
    Hparam.analytic_rd_curve = True
    Hparam.analytical_elbo = True
    Hparam.rd.num_betas = 50
    Hparam.rd.max_beta = 200
    Hparam.model_train.epochs = 100
    Hparam.rd.anneal_steps = 5000
    Hparam.svd = True
    Hparam.rd.target_dist = "joint_xz"
    Hparam.model_name = "vae_linear_fixed_var"
    Hparam.distortion_limit = 745.
    return Hparam


# ----------------------------------------------------------------------------
# RD BDMC
# ----------------------------------------------------------------------------


@register("BDMC_icml_vae10_rd")
def vae10_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_vae2_rd")
def vae2_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_vae100")
def __doesnotmatter():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 100
    Hparam.load_hparam_name = "icml_vae100"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam = BDMC_addon(Hparam)
    Hparam.beta_indexes = [20, 200, 1990]
    return Hparam


# ----------------------------------------------------------------------------
# Below are the reruns for second round
# ----------------------------------------------------------------------------
@register("icml_linear_vae100_rerun")
def vae100_linear_analytical_rerun():
    Hparam = vae100_linear_analytical()
    Hparam.train_first = False
    Hparam.load_hparam_name = "icml_linear_vae100"
    Hparam.step_sizes_target = "icml_linear_vae100"
    Hparam.random_seed = 7
    Hparam.distortion_limit = 745.
    return Hparam


@register("BDMC_icml_vae10_rd_rerun")
def vae10_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam = BDMC_addon(Hparam)
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_vae10_rd"
    return Hparam


@register("BDMC_icml_vae2_rd_rerun")
def vae2_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    Hparam = BDMC_addon(Hparam)
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_vae2_rd"
    return Hparam


@register("BDMC_icml_vae100_rerun")
def __doesnotmatter():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 100
    Hparam.load_hparam_name = "icml_vae100"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam = BDMC_addon(Hparam)
    Hparam.beta_indexes = [20, 200, 1990]
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_vae100"
    return Hparam


# ----------------------------------------------------------------------------
# mix prior rerun
# ----------------------------------------------------------------------------


@register("icml_vae10_mp_010_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.1
    Hparam.step_sizes_target = "icml_vae10_mp_010"
    return Hparam


@register("icml_vae10_mp_020_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.20
    Hparam.step_sizes_target = "icml_vae10_mp_020"
    return Hparam


@register("icml_vae10_mp_050_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.5
    Hparam.step_sizes_target = "icml_vae10_mp_050"
    return Hparam


@register("icml_vae10_mp_080_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.8
    Hparam.step_sizes_target = "icml_vae10_mp_080"
    return Hparam


@register("icml_vae10_mp_090_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.90
    Hparam.step_sizes_target = "icml_vae10_mp_090"
    return Hparam


@register("icml_vae10_mp_099_rerun")
def __doesnotmatter():
    Hparam = VAE_mix_prior_rerun()
    Hparam.mixture_weight = 0.99
    Hparam.step_sizes_target = "icml_vae10_mp_099"
    return Hparam


@register("icml_vae10_rd_rerun")
def vae10_rd():
    Hparam = VAE_RD_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    return Hparam


@register("icml_vae2_rd_rerun")
def vae2_rd():
    Hparam = VAE_RD_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    Hparam.step_sizes_target = "icml_vae2_rd"
    return Hparam


@register("icml_vae100_rerun")
def vae100_rd():
    Hparam = VAE_RD_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 100
    Hparam.load_hparam_name = "icml_vae100"
    Hparam.step_sizes_target = "icml_vae100"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


# ----------------------------------------------------------------------------
# BDMC
# ----------------------------------------------------------------------------


@register("icml_vae10_blur5_rerun")
def __dnm():
    Hparam = VAE_g_blur_rerun()
    Hparam.blur_std = 5
    Hparam.step_sizes_target = "icml_vae10_blur5"
    return Hparam


@register("icml_vae10_blur2_rerun")
def __dnm():
    Hparam = VAE_g_blur_rerun()
    Hparam.blur_std = 2
    Hparam.step_sizes_target = "icml_vae10_blur2"
    return Hparam


@register("icml_vae10_blur1_rerun")
def __dnm():
    Hparam = VAE_g_blur_rerun()
    Hparam.blur_std = 1
    Hparam.step_sizes_target = "icml_vae10_blur1"
    return Hparam


@register("icml_vae10_blur01_rerun")
def __dnm():
    Hparam = VAE_g_blur_rerun()
    Hparam.blur_std = 0.1
    Hparam.step_sizes_target = "icml_vae10_blur01"
    return Hparam


@register("icml_vae10_blur05_rerun")
def __dnm():
    Hparam = VAE_g_blur_rerun()
    Hparam.blur_std = 0.5
    Hparam.step_sizes_target = "icml_vae10_blur05"
    return Hparam


# # ----------------------------------------------------------------------------
# # random seed
# # ----------------------------------------------------------------------------


@register("icml_vae10_rerun_rs11")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 11
    return Hparam


@register("icml_vae10_rerun_rs12")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 12
    return Hparam


@register("icml_vae10_rerun_rs13")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 13
    return Hparam


@register("icml_vae10_rerun_rs14")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 14
    return Hparam


@register("icml_vae10_rerun_rs15")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 15
    return Hparam


@register("icml_vae10_rerun_rs16")
def vae10_simple_gvar_quick():
    Hparam = VAE_RD_rerun()
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "vae10_rd"
    Hparam.step_sizes_target = "icml_vae10_rd"
    Hparam.random_seed = 16
    return Hparam
