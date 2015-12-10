#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import logging
from numpy import RankWarning
from warnings import simplefilter

from .cannon import *
from .regularized import *
from . import (continuum, diagnostics, utils, vectorizer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)

# For debugging:
#handler.setFormatter(logging.Formatter(
#    "%(asctime)s [%(levelname)-8s] (%(name)s/%(lineno)d): %(message)s"))

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)

# Clean up the top-level namespace for this module.
del handler, logger, logging, RankWarning, simplefilter