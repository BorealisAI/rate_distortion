# Copyright (c) 2018-present, Royal Bank of Canada.
# Copyright (c) 2017 Yuhuai(Tony) Wu
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

# This code is based on
# https://github.com/tonywu95/eval_gen/blob/master/algorithms/ais.py

import numpy as np
_SCHEDULE = dict()


def register(name):

    def add_to_dict(fn):
        global _SCHEDULE
        _SCHEDULE[name] = fn
        return fn

    return add_to_dict


def get_schedule(task_params):
    return _SCHEDULE[task_params.temp_schedule](task_params)


@register("sigmoid")
def sigmoid_schedule(task_params):
    """The sigmoid schedule following: https://github.com/tonywu95/eval_gen/blob/master/algorithms/ais.py
          It's defined as:
          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),

    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    rad = 4 if task_params.sigmoid_rad is None else task_params.sigmoid_rad
    steps = task_params.anneal_steps
    if steps == 1:
        return [np.asarray(0.0), np.asarray(1.0)]
    t = np.linspace(-rad, rad, steps)
    sigm = 1. / (1. + np.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())


@register("linear")
def linear_schedule(task_params):
    return np.linspace(0., 1., task_params.anneal_steps)
