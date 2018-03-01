#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.2.93"

import logging
from numpy import RankWarning
from warnings import simplefilter

from .model import CannonModel
from . import (censoring, fitting, plot, utils, vectorizer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # TODO: Remove this when stable.

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)


def load_model(path, **kwargs):
    """
    Load a Cannon model from an existing filename, regardless of the kind of
    Cannon model sub-class.

    :param path:
        The path where the model has been saved. This saved model must include
        a labelled data set.
    """

    print("deprecated; use CannonModel.read") # TODO
    return CannonModel.read(path, **kwargs)


# Clean up the top-level namespace for this module.
del handler, logger, logging, RankWarning, simplefilter
