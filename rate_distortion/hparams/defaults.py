# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import register
from .hparam import Hparam as hp


def default_rd():
    """ rd """
    Hparam = hp()
    Hparam.n_chains = 40
    Hparam.anneal_steps = 60000
    Hparam.temp_schedule = "sigmoid"
    Hparam.leap_steps = 20
    Hparam.acceptance_prob = 0.65
    Hparam.n_batch = 1
    Hparam.batch_size = 50
    Hparam.rd_data_list = list(["test"])
    Hparam.target_dist = "joint_xz"
    Hparam.num_betas = 2000
    Hparam.max_beta = 36000
    Hparam.sigmoid_rad = 10
    return Hparam


def mnist():
    """ dataset """
    return hp(
        data_name="mnist",
        train_loader="mnist_train_valid",
        test_loader="mnist_test",
        eval_train_loader="mnist_eval_train",
        eval_test_loader="mnist_eval_test",
        input_dims=[1, 28, 28],
        input_vector_length=784)


def cifar10():
    """ dataset """
    return hp(
        data_name="cifar10",
        train_loader="cifar_train_valid",
        test_loader="CIFAR10_test",
        eval_train_loader="CIFAR10_eval_train",
        eval_test_loader="CIFAR10_eval_test",
        input_dims=[3, 32, 32],
        input_vector_length=3072,
        normalize=[0.5, 0.5])


def defualt_decoder():
    """ model_train """
    return hp(
        z_size=10,
        batch_size=100,  #used for VAE training, testing
        epochs=1000,
        x_var=0.03)


def default_experiment():
    Hparam = hp(
        output_root_dir="/home/rd_workspace",
        checkpoint_root_dir="/home/rd_workspace",
        data_dir="/home/rd_workspace/datasets",
        dataset=mnist(),
        cuda=True,
        verbose=True,
        random_seed=6,
        checkpointing_freq=None,
        chkt_step=-2,  #-2 means take the best, -1 means take the latest
        specific_model_path=None,
        model_train=defualt_decoder(),
        rd=default_rd(),
        plot_rd_curve=True,
        monitor_hmc=True,
        simplified_rate=True,
        mixed_precision=True)
    return Hparam


def mnist_fid_default():
    Hparam = default_experiment()
    Hparam.cnn_model_name = "mnist_cnn"
    Hparam.representation = hp()
    Hparam.representation.test_batch_size = 1000
    Hparam.representation.batch_size = 64
    Hparam.representation.epochs = 10
    Hparam.representation.lr = 0.01
    Hparam.representation.momentum = 0.5
    Hparam.representation.log_interval = 10
    Hparam.rep_train_first = False
    Hparam.load_rep = "cnn_mnist"
    Hparam.rd.target_dist = "MNIST_fid"
    Hparam.chkt_step = -2  #-2 means take the best, -1 means take the latest
    Hparam.original_experiment = True
    Hparam.rd.max_beta = 0.33
    Hparam.rd.anneal_steps = 100000
    return Hparam


def BDMC_addon(Hparam):
    """ 
    Turning an AIS run into a BDMC run.
     """
    Hparam.BDMC_points = 3
    Hparam.rd.rd_data_list = list(["simulate"])
    Hparam.simulated_data = "simulate_default"
    Hparam.train_first = False
    Hparam.beta_indexes = [20, 48, 1990]
    return Hparam
