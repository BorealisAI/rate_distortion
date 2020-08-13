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
from .registry import register
from torch.nn import functional as F


@register("mnist_cnn")
def get_vae(hparams):

    class Net(nn.Module):

        def __init__(self, hparams):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = self.fc1(x)
            return F.log_softmax(x, dim=1)

        def get_rep_1(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(self.conv1(x))
            return x.view(x.size()[0], -1)

        def get_rep_2(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            return x.view(x.size()[0], -1)

    return Net(hparams)
