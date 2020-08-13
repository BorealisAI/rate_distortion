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
def aae_experiment():
    Hparam = default_experiment()
    Hparam.train_first = False
    Hparam.model_name = "aae_deep"
    Hparam.load_hparam_name = "aae"
    Hparam.original_experiment = True
    Hparam.specific_model_path = True
    Hparam.rd.target_dist = "MSE"
    Hparam.group_list = ["icml", "AAEs"]
    return Hparam


def aae_experiment_rerun():
    Hparam = aae_experiment()
    Hparam.random_seed = 7
    return Hparam


# ----------------------------------------------------------------------------
# AAE RD
# ----------------------------------------------------------------------------
@register("icml_aae2")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    return Hparam


@register("icml_aae10")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    return Hparam


@register("icml_aae100")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 100
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam.group_list = None
    Hparam.weights_name = "aae_h100_stoch_reg01_auto01_test_weights_numpy.npz"
    return Hparam


# ----------------------------------------------------------------------------
# BDMC AAE RD
# ----------------------------------------------------------------------------
@register("BDMC_icml_aae2")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_aae10")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_aae100")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.rd.max_beta = 3333
    Hparam.model_train.z_size = 100
    Hparam.rd.anneal_steps = 1600000
    Hparam.weights_name = "aae_h100_stoch_reg01_auto01_test_weights_numpy.npz"
    Hparam = BDMC_addon(Hparam)
    Hparam.beta_indexes = [20, 200, 1990]
    return Hparam


# ----------------------------------------------------------------------------
# BDMC AAE RD rerun
# ----------------------------------------------------------------------------
@register("BDMC_icml_aae2_rerun")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "BDMC_icml_aae2"
    Hparam.random_seed = 7
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_aae10_rerun")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "BDMC_icml_aae10"
    Hparam.random_seed = 7
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_aae100_rerun")
def __doesnotmatter():
    Hparam = aae_experiment()
    Hparam.rd.max_beta = 3333
    Hparam.model_train.z_size = 100
    Hparam.rd.anneal_steps = 1600000
    Hparam.weights_name = "aae_h100_stoch_reg01_auto01_test_weights_numpy.npz"
    Hparam = BDMC_addon(Hparam)
    Hparam.step_sizes_target = "BDMC_icml_aae100"
    Hparam.random_seed = 7
    Hparam.beta_indexes = [20, 200, 1990]
    return Hparam


# ----------------------------------------------------------------------------
# AAE RD rerun
# ----------------------------------------------------------------------------
@register("icml_aae2_rerun")
def __doesnotmatter():
    Hparam = aae_experiment_rerun()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "icml_aae2"
    return Hparam


@register("icml_aae10_rerun")
def __doesnotmatter():
    Hparam = aae_experiment_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "icml_aae10"
    return Hparam


@register("icml_aae100_rerun")
def __doesnotmatter():
    Hparam = aae_experiment_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 100
    Hparam.weights_name = "aae_h100_stoch_reg01_auto01_test_weights_numpy.npz"
    Hparam.step_sizes_target = "icml_aae100"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam
