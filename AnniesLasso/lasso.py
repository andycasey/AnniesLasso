#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A compressed sensing edition of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LassoCannonModel"]

import logging
import numpy as np
import scipy.optimize as op

from . import (model, utils)

logger = logging.getLogger(__name__)


class LassoCannonModel(model.BaseCannonModel):
    """
    A compressed sensing edition of The Cannon model for the estimation of 
    arbitrary stellar labels with regularized complexity.

    :param labels:
        A table with columns as labels, and stars as rows.

    :type labels:
        :class:`~astropy.table.Table` or numpy structured array

    :param fluxes:
        An array of fluxes for stars in the training set, given as shape
        `(num_stars, num_pixels)`. The `num_stars` should match the number of
        rows in `labels`.

    :type fluxes:
        :class:`np.ndarray`

    :param flux_uncertainties:
        An array of 1-sigma flux uncertainties for stars in the training set,
        The shape of the `flux_uncertainties` should match `fluxes`. 

    :type flux_uncertainties:
        :class:`np.ndarray`

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have length `num_pixels`.

    :param live_dangerously: [optional]
        If enabled then no checks will be made on the label names, prohibiting
        the user to input human-readable forms of the label vector.
    """

    _data_attributes = ["training_labels", "training_fluxes",
        "training_flux_uncertainties"]
    _trained_attributes = ["_label_vector", "_coefficients", "_scatter"]
    _forbidden_label_characters = "^*"

    def __init__(self, *args, **kwargs):
        super(LassoCannonModel, self).__init__(*args, **kwargs)


    # This class has additional attributes.
    # label_complexity?
    # regularization_behaviour?
    # etc...

    @model.requires_label_vector
    def train(self, **kwargs):
        raise NotImplementedError


    @model.requires_training_wheels
    def predict(self, labels=None, **labels_as_kwargs):
        raise NotImplementedError


    @model.requires_training_wheels
    def fit(self, fluxes, flux_uncertainties, **kwargs):
        raise NotImplementedError


