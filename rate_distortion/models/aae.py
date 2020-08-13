# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from ..utils.vae_utils import log_normal, log_normal_likelihood
from ..utils.computation_utils import singleton_repeat
from .registry import register


@register("aae_deep")
def get_vae(hparams):
    """
    This function will return the decoder of the 
    Adversarial Autoencoder used in the paper 
    """

    class VAE(nn.Module):

        def __init__(self, hparams):
            super(VAE, self).__init__()
            # decoder
            self.de1 = nn.Linear(hparams.model_train.z_size, 1024)
            self.de2 = nn.Linear(1024, 1024)
            self.de3 = nn.Linear(1024, 1024)
            self.de_mean = nn.Linear(1024, hparams.dataset.input_vector_length)
            self.x_logvar = nn.Parameter(
                torch.log(
                    torch.tensor(
                        hparams.model_train.x_var, dtype=hparams.tensor_type)),
                requires_grad=False)
            self.observation_log_likelihood_fn = log_normal_likelihood
            self.act = nn.ReLU()

        def decode(self, z):

            h1 = self.act(self.de1(z))
            h2 = self.act(self.de2(h1))
            h3 = self.act(self.de3(h2))

            mean = torch.sigmoid(self.de_mean(h3))
            return mean, self.x_logvar

    return VAE(hparams)
