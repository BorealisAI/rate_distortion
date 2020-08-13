# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

_LOADERS = dict()


def register(name):

    def add_to_dict(fn):
        global _LOADERS
        _LOADERS[name] = fn
        return fn

    return add_to_dict


def get_loader(dataset_name, batch_size, hparams):

    return _LOADERS[dataset_name](batch_size, hparams)
