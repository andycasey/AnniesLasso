#!/usr/bin/env python
# -*- coding: utf-8 -*-

__version__ = "0.1.0"

import logging
from numpy import RankWarning
from warnings import simplefilter
from sys import version_info

from .cannon import *
from .regularized import *
from . import (censoring, continuum, diagnostics, utils, vectorizer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # TODO: Remove this when stable.

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s"))
logger.addHandler(handler)

# For debugging:
#handler.setFormatter(logging.Formatter(
#    "%(asctime)s [%(levelname)-8s] (%(name)s/%(lineno)d): %(message)s"))

simplefilter("ignore", RankWarning)
simplefilter("ignore", RuntimeWarning)


def load_model(filename, **kwargs):
    """
    Load a Cannon model from an existing filename, regardless of the kind of
    Cannon model sub-class.

    :param filename:
        The path where the model has been saved. This saved model must include
        a labelled data set.
    """
    from six.moves import cPickle as pickle # I know.

    encodings = ("utf-8", "latin-1")
    for encoding in encodings:
        kwds = {"encoding": encoding} if version_info[0] >= 3 else {}
        try:
            with open(filename, "rb") as fp:        
                contents = pickle.load(fp, **kwds)

        except UnicodeDecodeError:
            if encoding == encodings:
                raise

    class_factory = contents["metadata"]["model_name"]
    if not class_factory.endswith("CannonModel"):
        raise TypeError("Cannon model factory class '{}' not recognised".format(
            class_factory))

    _class = eval(class_factory)
    has_data = (contents["metadata"]["data_attributes"][0] in contents)
    if not has_data:
        contents.update({
            "labelled_set": {},
            "normalized_flux": [],
            "normalized_ivar": []
        })

    # Initiate the class.
    kwds = { "verify": False }
    kwds.update(**kwargs)

    model = _class(
        *[contents.get(attr, None) for attr in contents["metadata"]["data_attributes"]],
        **kwds)

    # Update information.
    model._metadata.update(contents["metadata"].get("metadata", {}))
    attributes = contents["metadata"]["descriptive_attributes"] \
               + contents["metadata"]["trained_attributes"]

    for attribute in attributes:
        setattr(model, "_{}".format(attribute), contents[attribute])

    # Censors must be correctly linked.
    model._censors = censoring.CensorsDict(model, model._censors)

    return model


# Clean up the top-level namespace for this module.
del handler, logger, logging, RankWarning, simplefilter
