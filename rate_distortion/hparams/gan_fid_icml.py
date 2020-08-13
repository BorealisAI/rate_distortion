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
def gan_experiment():
    Hparam = mnist_fid_default()
    Hparam.model_name = "gan"
    return Hparam


def GAN_MNIST():
    Hparam = gan_experiment()
    return Hparam


def GAN_MNIST_rerun():
    Hparam = GAN_MNIST()
    Hparam.random_seed = 7
    return Hparam


# ----------------------------------------------------------------------------
# Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("fid_icml_deep_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x43a407bc_LT/LT_delta-0x51a60939_base-0x43a407bc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


@register("fid_icml_deep_gan10_GP_test")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


@register("fid_icml_deep_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


@register("fid_icml_deep_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


# ----------------------------------------------------------------------------
# Shallow GAN MNIST
# ----------------------------------------------------------------------------
@register("fid_icml_shallow_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x3e3c07fc_LT/LT_delta-0x4f6e080c_base-0x3e3c07fc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


@register("fid_icml_shallow_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x520309ca_LT/LT_delta-0x4f6e080c_base-0x520309ca_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


@register("fid_icml_shallow_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x445c0795_LT/LT_delta-0x395a06e9_base-0x445c0795_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


# ----------------------------------------------------------------------------
# BDMC  Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("BDMC_fid_icml_deep_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x43a407bc_LT/LT_delta-0x51a60939_base-0x43a407bc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_fid_icml_deep_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_fid_icml_deep_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


# ----------------------------------------------------------------------------
# BDMC Shallow GAN MNIST
# ----------------------------------------------------------------------------
@register("BDMC_fid_icml_shallow_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x3e3c07fc_LT/LT_delta-0x4f6e080c_base-0x3e3c07fc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_fid_icml_shallow_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x520309ca_LT/LT_delta-0x4f6e080c_base-0x520309ca_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_fid_icml_shallow_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x445c0795_LT/LT_delta-0x395a06e9_base-0x445c0795_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


# ----------------------------------------------------------------------------
# Below are re-runs
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ----------------------------------------------------------------------------
# Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("fid_icml_deep_gan5_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x43a407bc_LT/LT_delta-0x51a60939_base-0x43a407bc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "fid_icml_deep_gan5_GP"
    return Hparam


@register("fid_icml_deep_gan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "fid_icml_deep_gan10_GP"
    return Hparam


@register("fid_icml_deep_gan2_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "fid_icml_deep_gan2_GP"
    return Hparam


# ----------------------------------------------------------------------------
# Shallow GAN MNIST
# ----------------------------------------------------------------------------
@register("fid_icml_shallow_gan5_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x3e3c07fc_LT/LT_delta-0x4f6e080c_base-0x3e3c07fc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "fid_icml_shallow_gan5_GP"
    return Hparam


@register("fid_icml_shallow_gan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x520309ca_LT/LT_delta-0x4f6e080c_base-0x520309ca_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "fid_icml_shallow_gan10_GP"
    return Hparam


@register("fid_icml_shallow_gan2_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x445c0795_LT/LT_delta-0x395a06e9_base-0x445c0795_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "fid_icml_shallow_gan2_GP"
    return Hparam
