# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

from ..models.registry import get_model
from ..utils.experiment_utils import note_taking
import numpy as np
import torch.nn as nn
import torch


def load_aae_weights(model, aae_weights, hparams):
    """ 
    Load numpy weights for the aae used in the paper, 
    which was originally trained in tensorflow. 
    """
    model.de1.weight = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h1/w_dense:0"].transpose()).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de1.bias = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h1/b_dense:0"]).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de1.requires_grad = False
    model.de2.weight = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h2/w_dense:0"].transpose()).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de2.bias = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h2/b_dense:0"]).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de2.requires_grad = False

    model.de3.weight = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h3/w_dense:0"].transpose()).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de3.bias = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h3/b_dense:0"]).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de_mean.requires_grad = False

    model.de_mean.weight = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h4/w_dense:0"].transpose()).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de_mean.bias = nn.Parameter(
        torch.from_numpy(aae_weights["decoder_h4/b_dense:0"]).to(
            device=hparams.device, dtype=hparams.tensor_type))
    model.de_mean.requires_grad = False


def aae_bridge(hparams):
    weight_path = hparams.messenger.checkpoint_dir + hparams.weights_name
    aae_weights = np.load(weight_path)
    model = get_model(hparams)
    load_aae_weights(model, aae_weights, hparams)

    return model.to(hparams.device)