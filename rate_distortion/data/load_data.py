# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from .registry import get_loader
from ..utils.experiment_utils import note_taking
from ..models.simulate import get_simulate_data
import torch


def load_training_data(hparams):
    train_loader, test_loader = get_loader(
        hparams.dataset.train_loader, hparams.model_train.batch_size, hparams)

    return train_loader, test_loader


def load_rd_data(model, hparams, data):
    """ 
    To speed up rate-distorion computation, only supports 1-batch. 
     """

    if data == "train":
        rd_loader = get_loader(hparams.dataset.eval_train_loader,
                               hparams.rd.batch_size, hparams)

    elif data == "test":
        rd_loader = get_loader(hparams.dataset.eval_test_loader,
                               hparams.rd.batch_size, hparams)

    elif data == "simulate":

        if hparams.rd.simulate_dir is None:
            rd_loader = get_simulate_data(
                model,
                hparams.rd.batch_size,
                hparams.rd.n_batch,
                hparams,
                rd=True)
        else:
            # Assume number of batch is 1 to maximize GPU efficiency.
            x_path = hparams.rd.simulate_dir + "rd_simulated_x.pt"
            z_path = hparams.rd.simulate_dir + "rd_simulated_z.pt"
            x_tensor = torch.load(x_path)
            z_tensor = torch.load(z_path)
            note_taking("Simulated rd data is loaded onto {}".format(
                x_tensor.device))
            rd_loader = list()

            if int(x_tensor.size()[0]) < hparams.rd.batch_size:
                note_taking(
                    "WARNING: failed to load {} # data from {} saved data, setting batch size to {} instead"
                    .format(hparams.rd.batch_size, int(x_tensor.size()[0]),
                            int(x_tensor.size()[0])))
                hparams.rd.batch_size = int(x_tensor.size()[0])

            x = x_tensor[:hparams.rd.batch_size, :]
            z = z_tensor[:hparams.rd.batch_size, :]
            rd_loader.append((x, z))

    return rd_loader


def load_simulate_data(model, hparams, beta, simulate_dir=None):
    """ 
    To speed up computation, only supports 1-batch. 
     """

    simulate_loader = get_simulate_data(
        model,
        hparams.rd.batch_size,
        hparams.rd.n_batch,
        hparams,
        rd=True,
        beta=beta)

    return simulate_loader
