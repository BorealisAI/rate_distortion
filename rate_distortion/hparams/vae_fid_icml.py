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


def vae_experiment_fid_icml():
    Hparam = mnist_fid_default()
    Hparam.model_name = "deep_vae"
    return Hparam


def VAE_RD():
    Hparam = vae_experiment_fid_icml()
    Hparam.model_name = "deep_vae"
    Hparam.group_list = ["icml", "VAEs", "RD"]
    return Hparam


def VAE_RD_rerun():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.random_seed = 7
    return Hparam


@register("fid_icml_vae10_rd")
def vae10_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    return Hparam


@register("fid_icml_vae2_rd")
def vae2_rd():
    Hparam = VAE_RD()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    return Hparam


@register("fid_icml_vae10_rd_rerun")
def vae10_rd():
    Hparam = VAE_RD_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 10
    Hparam.load_hparam_name = "icml_vae10_rd"
    Hparam.step_sizes_target = "fid_icml_vae10_rd"
    return Hparam


@register("fid_icml_vae2_rd_rerun")
def vae2_rd():
    Hparam = VAE_RD_rerun()
    Hparam.train_first = False
    Hparam.model_train.z_size = 2
    Hparam.load_hparam_name = "icml_vae2_rd"
    Hparam.step_sizes_target = "fid_icml_vae2_rd"
    return Hparam
