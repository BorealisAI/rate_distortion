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
    Hparam = default_experiment()
    Hparam.model_name = "gan"
    Hparam.original_experiment = True
    Hparam.rd.target_dist = "MSE"
    return Hparam


def GAN_MNIST():
    Hparam = gan_experiment()
    return Hparam


def GAN_CIFAR():
    Hparam = gan_experiment()
    Hparam.dataset = cifar10()
    Hparam.rd.max_beta = 3333
    Hparam.group_list = ["icml", "GANs", "CIFAR10"]

    return Hparam


def GAN_MNIST_rerun():
    Hparam = GAN_MNIST()
    Hparam.random_seed = 7
    Hparam.group_list = ["icml", "GANs"]
    return Hparam


def GAN_CIFAR_rerun():
    Hparam = GAN_CIFAR()
    Hparam.random_seed = 7
    Hparam.group_list = ["icml", "GANs", "CIFAR10"]
    return Hparam


# ----------------------------------------------------------------------------
# Deep GAN CIFAR RD
# ----------------------------------------------------------------------------
@register("icml_CIFAR10_dcgan5_BRE")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x47890899_LT/LT_delta-0x406a0783_base-0x47890899_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan10_BRE")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x435c0841_LT/LT_delta-0x406a0783_base-0x435c0841_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan100_BRE")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x4ea60959_LT/LT_delta-0x406a0783_base-0x4ea60959_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan5_SN")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x57cb0a10_LT/LT_delta-0x450c084a_base-0x57cb0a10_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan10_SN")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x476d0977_LT/LT_delta-0x450c084a_base-0x476d0977_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan100_SN")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x501c09f0_LT/LT_delta-0x450c084a_base-0x501c09f0_LT_netG_epoch_63.pth"
    return Hparam


@register("icml_CIFAR10_dcgan5_GP")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x37ab0672_LT/LT_delta-0x4f6e080c_base-0x37ab0672_LT_netG_epoch_128.pth"
    return Hparam


@register("icml_CIFAR10_dcgan10_GP")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x5068093f_LT/LT_delta-0x4f6e080c_base-0x5068093f_LT_netG_epoch_128.pth"
    return Hparam


@register("BDMC_icml_CIFAR10_dcgan10_GP")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x5068093f_LT/LT_delta-0x4f6e080c_base-0x5068093f_LT_netG_epoch_128.pth"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("icml_CIFAR10_dcgan100_GP")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x4918090c_LT/LT_delta-0x4f6e080c_base-0x4918090c_LT_netG_epoch_128.pth"
    return Hparam


@register("BDMC_icml_CIFAR10_dcgan100_GP")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x4918090c_LT/LT_delta-0x4f6e080c_base-0x4918090c_LT_netG_epoch_128.pth"
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("icml_CIFAR10_dcgan5_V")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x38930725_LT/LT_delta-0x4f6e080c_base-0x38930725_LT_netG_epoch_128.pth"
    return Hparam


@register("icml_CIFAR10_dcgan10_V")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x3b4c07f5_LT/LT_delta-0x4f6e080c_base-0x3b4c07f5_LT_netG_epoch_128.pth"
    return Hparam


@register("icml_CIFAR10_dcgan100_V")
def __doesnotmatter():
    Hparam = GAN_CIFAR()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x5c180a3b_LT/LT_delta-0x4f6e080c_base-0x5c180a3b_LT_netG_epoch_128.pth"
    return Hparam


# ----------------------------------------------------------------------------
# Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("icml_deep_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x43a407bc_LT/LT_delta-0x51a60939_base-0x43a407bc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


@register("icml_deep_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


@register("icml_deep_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    return Hparam


# ----------------------------------------------------------------------------
# Shallow GAN MNIST
# ----------------------------------------------------------------------------
@register("icml_shallow_gan5_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x3e3c07fc_LT/LT_delta-0x4f6e080c_base-0x3e3c07fc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


@register("icml_shallow_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x520309ca_LT/LT_delta-0x4f6e080c_base-0x520309ca_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


@register("icml_shallow_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x445c0795_LT/LT_delta-0x395a06e9_base-0x445c0795_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    return Hparam


@register("icml_shallow_gan100_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x3c6f0702_LT/LT_delta-0x395a06e9_base-0x3c6f0702_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


@register("icml_deep_gan100_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_debug4_mnist_gp_LT_delta-0x450c084a_base-0x461907a5_LT/LT_delta-0x450c084a_base-0x461907a5_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


@register("BDMC_icml_shallow_gan100_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x3c6f0702_LT/LT_delta-0x395a06e9_base-0x3c6f0702_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_deep_gan100_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_debug4_mnist_gp_LT_delta-0x450c084a_base-0x461907a5_LT/LT_delta-0x450c084a_base-0x461907a5_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam = BDMC_addon(Hparam)
    Hparam.beta_indexes = [20, 200, 1990]
    return Hparam


# ----------------------------------------------------------------------------
# BDMC  Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("BDMC_icml_deep_gan10_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


@register("BDMC_icml_deep_gan2_GP")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    return Hparam


# ----------------------------------------------------------------------------
# Below are re-runs
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@register("BDMC_icml_deep_gan100_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_debug4_mnist_gp_LT_delta-0x450c084a_base-0x461907a5_LT/LT_delta-0x450c084a_base-0x461907a5_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    Hparam = BDMC_addon(Hparam)
    Hparam.beta_indexes = [20, 200, 1990]
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_deep_gan100_GP"
    return Hparam


@register("BDMC_icml_deep_gan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_deep_gan10_GP"
    return Hparam


@register("BDMC_icml_deep_gan2_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam = BDMC_addon(Hparam)
    Hparam.random_seed = 7
    Hparam.step_sizes_target = "BDMC_icml_deep_gan2_GP"
    return Hparam


# ----------------------------------------------------------------------------
# Deep GAN MNIST RD
# ----------------------------------------------------------------------------


@register("icml_deep_gan5_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x43a407bc_LT/LT_delta-0x51a60939_base-0x43a407bc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan5_GP"
    return Hparam


@register("icml_deep_gan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_gp_mlp-big_LT_delta-0x51a60939_base-0x4f6d0964_LT/LT_delta-0x51a60939_base-0x4f6d0964_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan10_GP"
    return Hparam


@register("icml_deep_gan5_v_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_vanilla_mlp-big_LT_delta-0x51a60939_base-0x448808df_LT/LT_delta-0x51a60939_base-0x448808df_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan5_v"
    return Hparam


@register("icml_deep_gan10_v_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may17_vanilla_mlp-big_LT_delta-0x51a60939_base-0x3afd080f_LT/LT_delta-0x51a60939_base-0x3afd080f_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan10_v"
    return Hparam


@register("icml_deep_gan100_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_debug4_mnist_gp_LT_delta-0x450c084a_base-0x461907a5_LT/LT_delta-0x450c084a_base-0x461907a5_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan100_GP"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


@register("icml_deep_gan2_v_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_vanilla_mlp-big_LT_delta-0x517f09d5_base-0x4038085a_LT/LT_delta-0x517f09d5_base-0x4038085a_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan2_v"
    return Hparam


@register("icml_deep_gan2_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final2_gp_mlp-big_LT_delta-0x517f09d5_base-0x32dc06b8_LT/LT_delta-0x517f09d5_base-0x32dc06b8_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_deep"]
    Hparam.step_sizes_target = "icml_deep_gan2_GP"
    return Hparam


# ----------------------------------------------------------------------------
# Shallow GAN MNIST
# ----------------------------------------------------------------------------
@register("icml_shallow_gan5_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x3e3c07fc_LT/LT_delta-0x4f6e080c_base-0x3e3c07fc_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan5_GP"
    return Hparam


@register("icml_shallow_gan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_gp_mlp-small_LT_delta-0x4f6e080c_base-0x520309ca_LT/LT_delta-0x4f6e080c_base-0x520309ca_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan10_GP"
    return Hparam


@register("icml_shallow_gan5_V_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_vanilla_mlp-small_LT_delta-0x4f6e080c_base-0x432b071c_LT/LT_delta-0x4f6e080c_base-0x432b071c_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan5_V"
    return Hparam


@register("icml_shallow_gan10_V_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_may16_vanilla_mlp-small_LT_delta-0x4f6e080c_base-0x3379057d_LT/LT_delta-0x4f6e080c_base-0x3379057d_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan10_V"
    return Hparam


@register("icml_shallow_gan2_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x445c0795_LT/LT_delta-0x395a06e9_base-0x445c0795_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan2_GP"
    return Hparam


@register("icml_shallow_gan100_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_gp_mlp-small_LT_delta-0x395a06e9_base-0x3c6f0702_LT/LT_delta-0x395a06e9_base-0x3c6f0702_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan100_GP"
    Hparam.rd.max_beta = 3333
    Hparam.rd.anneal_steps = 1600000
    return Hparam


@register("icml_shallow_gan2_v_rerun")
def __doesnotmatter():
    Hparam = GAN_MNIST_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "mnist_final_vanilla_mlp-small_LT_delta-0x395a06e9_base-0x3545061e_LT/LT_delta-0x395a06e9_base-0x3545061e_LT_netG_epoch_106.pth"
    Hparam.group_list = ["icml", "GANs", "MNIST_shallow"]
    Hparam.step_sizes_target = "icml_shallow_gan2_v"
    return Hparam


# ----------------------------------------------------------------------------
# Deep GAN CIFAR RD
# ----------------------------------------------------------------------------
@register("icml_CIFAR10_dcgan5_BRE_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x47890899_LT/LT_delta-0x406a0783_base-0x47890899_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan5_BRE"
    return Hparam


@register("icml_CIFAR10_dcgan10_BRE_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x435c0841_LT/LT_delta-0x406a0783_base-0x435c0841_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan10_BRE"
    return Hparam


@register("icml_CIFAR10_dcgan100_BRE_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may18_bre_dcgan_LT_delta-0x406a0783_base-0x4ea60959_LT/LT_delta-0x406a0783_base-0x4ea60959_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan100_BRE"
    return Hparam


@register("icml_CIFAR10_dcgan5_SN_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x57cb0a10_LT/LT_delta-0x450c084a_base-0x57cb0a10_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan5_SN"
    return Hparam


@register("icml_CIFAR10_dcgan10_SN_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x476d0977_LT/LT_delta-0x450c084a_base-0x476d0977_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan10_SN"
    return Hparam


@register("icml_CIFAR10_dcgan100_SN_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may21_spectral_dcgan_LT_delta-0x450c084a_base-0x501c09f0_LT/LT_delta-0x450c084a_base-0x501c09f0_LT_netG_epoch_63.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan100_SN"
    return Hparam


@register("icml_CIFAR10_dcgan5_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x37ab0672_LT/LT_delta-0x4f6e080c_base-0x37ab0672_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan5_GP"
    return Hparam


@register("icml_CIFAR10_dcgan10_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x5068093f_LT/LT_delta-0x4f6e080c_base-0x5068093f_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan10_GP"
    return Hparam


@register("icml_CIFAR10_dcgan100_GP_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_gp_dcgan_LT_delta-0x4f6e080c_base-0x4918090c_LT/LT_delta-0x4f6e080c_base-0x4918090c_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan100_GP"
    return Hparam


@register("icml_CIFAR10_dcgan5_V_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x38930725_LT/LT_delta-0x4f6e080c_base-0x38930725_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan5_V"
    return Hparam


@register("icml_CIFAR10_dcgan10_V_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x3b4c07f5_LT/LT_delta-0x4f6e080c_base-0x3b4c07f5_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan10_V"
    return Hparam


@register("icml_CIFAR10_dcgan100_V_rerun")
def __doesnotmatter():
    Hparam = GAN_CIFAR_rerun()
    Hparam.specific_model_path = Hparam.output_root_dir + "/checkpoints/gans/results/" + "cifar10_may16_vanilla_dcgan_LT_delta-0x4f6e080c_base-0x5c180a3b_LT/LT_delta-0x4f6e080c_base-0x5c180a3b_LT_netG_epoch_128.pth"
    Hparam.step_sizes_target = "icml_CIFAR10_dcgan100_V"
    return Hparam
