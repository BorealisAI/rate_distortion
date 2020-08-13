# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

_MODEL = dict()


def register(name):

    def add_to_dict(fn):
        global _MODEL
        _MODEL[name] = fn
        return fn

    return add_to_dict


def get_model(hparams):
    return _MODEL[hparams.model_name](hparams)


def get_cnn_model(hparams):
    return _MODEL[hparams.cnn_model_name](hparams)
