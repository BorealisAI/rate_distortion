# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

import numpy as np
import torch
from .experiment_utils import note_taking


def singleton_repeat(x, n):
    """ 
    Repeat a batch of data n times. 
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back. 
    So that later when reshaping, it's guranteed to follow the same tensor convention. 
     """
    if n == 1:
        return x
    else:
        singleton_x = torch.unsqueeze(x, 0)
        repeated_x = singleton_x.repeat(n, 1, 1)
        return repeated_x.view(-1, x.size()[-1])


def normalized_weights_test(normalized_logws, n_chains, batch_size,
                            unnormalized_logws):
    test_logws_sum = torch.sum(normalized_logws, dim=0, keepdim=True)
    note_taking("Testing whether normalized_logws sum up to 1: {}".format(
        test_logws_sum))


def normalize_logws(logws, batch_size, n_chains, hparams, dim=0):

    if hparams.mixed_precision:
        rearanged_logws = logws.view(n_chains, batch_size, -1).to(
            device=hparams.device, dtype=torch.float64)
    else:
        rearanged_logws = logws.view(n_chains, batch_size, -1)

    chain_sum_logws = torch.logsumexp(rearanged_logws, dim=0, keepdim=True)
    normalized_log_w = rearanged_logws - chain_sum_logws
    normalized_w = torch.exp(normalized_log_w)
    normalizing_constants_logws = chain_sum_logws - torch.log(
        torch.tensor(n_chains).to(hparams.tensor_type))

    if hparams.mixed_precision:
        normalizing_constants_logws = normalizing_constants_logws.to(
            device=hparams.device, dtype=hparams.tensor_type)
        normalized_w = normalized_w.to(
            device=hparams.device, dtype=hparams.tensor_type)

    return normalized_w, normalizing_constants_logws
