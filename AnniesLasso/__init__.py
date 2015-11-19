#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__version__ = "0.1.0"

import logging
from numpy import RankWarning
from warnings import simplefilter

from .cannon import *
from .model import *
from . import utils


logger = logging.getLogger("cannon")
logger.setLevel(logging.CRITICAL)

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)