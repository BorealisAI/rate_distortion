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
def aae_experiment_fid_icml():
    Hparam = mnist_fid_default()
    Hparam.train_first = False
    Hparam.model_name = "aae_deep"
    Hparam.load_hparam_name = "aae"
    Hparam.group_list = ["icml", "AAEs"]
    return Hparam


def aae_experiment_fid_icml_rerun():
    Hparam = aae_experiment_fid_icml()
    Hparam.random_seed = 7
    return Hparam


# ----------------------------------------------------------------------------
# AAE RD
# ----------------------------------------------------------------------------
@register("fid_icml_aae2")
def __doesnotmatter():
    Hparam = aae_experiment_fid_icml()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    return Hparam


@register("fid_icml_aae10")
def __doesnotmatter():
    Hparam = aae_experiment_fid_icml()
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    return Hparam


# ----------------------------------------------------------------------------
# rerun AAE RD
# ----------------------------------------------------------------------------
@register("fid_icml_aae2_rerun")
def __doesnotmatter():
    Hparam = aae_experiment_fid_icml_rerun()
    Hparam.model_train.z_size = 2
    Hparam.weights_name = "aae_h2_det_reg1_auto001_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "fid_icml_aae2"
    return Hparam


@register("fid_icml_aae10_rerun")
def __doesnotmatter():
    Hparam = aae_experiment_fid_icml_rerun()
    Hparam.model_train.z_size = 10
    Hparam.weights_name = "aae_h10_det_reg1_auto01_annealed_epoch5000_weights_numpy.npz"
    Hparam.step_sizes_target = "fid_icml_aae10"
    return Hparam
