#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A dictionary sub-class to deal with wavelength censoring.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CensorsDict"]

import logging
import numpy as np

logger = logging.getLogger(__name__)


# A custom class is needed because if you have a dictionary attribute that is
# a @property of a class, it can be updated by `model.censors["key"] = ...`
# which will bypass the @property.setter function of that class.

class CensorsDict(dict):

    def __init__(self, model, *args, **kwargs):
        self.model = model
        return None

    def __setitem__(self, label_name, value):
        """
        Update one of the entries of the wavelength censoring dictionary.

        :param label_name:
            The name of the label to apply the censoring onto.

        :param value:
            A boolean mask the same length as the `model.dispersion`, or a set
            of (start, end) ranges to censor (*exclude*).
        """

        if label_name not in self.model.vectorizer.label_names:
            logger.warn(
                "Ignoring unrecognized label name '{}' in the wavelength "
                "censoring description".format(label_name))
            return None

        value = np.atleast_2d(value)
        if value.size == self.model.dispersion.size:
            # A mask was given. Ensure it is boolean.
            value = value.flatten().astype(bool)

        elif len(value.shape) == 2 and value.shape[1] == 2:
            # Ranges specified. Generate boolean mask.
            mask = np.ones(self.model.dispersion.size, dtype=bool)
            for start, end in value:
                exc = (end >= self.model.dispersion) \
                    * (self.model.dispersion >= start)
                mask[exc] = False

            value = mask

        else:
            raise ValueError("cannot interpret censoring mask for label '{}'"\
                .format(label_name))

        dict.__setitem__(self, label_name, value)