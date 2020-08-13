# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3

__all__ = [
    "cnns",
    "vaes",
    "aae",
    "gans",
    "user_models",
]
from .cnns import *
from .vaes import *
from .aae import *
from .gans import *
from .user_models import *
"""
If you need to register models in more files, add them here.
"""