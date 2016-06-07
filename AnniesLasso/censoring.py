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

class CensorsDict(dict):

    def __init__(self, model, *args, **kwargs):
        """
        A dictionary sub-class that allows for wavelength censoring masks to be
        applied to Cannon models.

        :param model:
            The Cannon model for which this censored dictionary will be applied.

        Note:   This custom class is necessary because if you have a dictionary
                attribute that is a `@property` of a class then it can be
                updated directly by `model.censors["label"] = ...` which will
                bypass the `@property.setter` method of the model class.
        """
        super(CensorsDict, self).__init__(*args, **kwargs)
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

        if  self.model.vectorizer is not None \
        and label_name not in self.model.vectorizer.label_names:
            logger.warn(
                "Ignoring unrecognized label name '{}' in the wavelength "
                "censoring description".format(label_name))

        value = np.atleast_2d(value)
        if value.size == self.model.dispersion.size:
            # A mask was given. Ensure it is boolean.
            if not np.all(np.isfinite(value)):
                raise ValueError("non-finite values given as a boolean mask")

            value = value.flatten().astype(bool)

        elif len(value.shape) == 2 and value.shape[1] == 2:
            # Ranges specified. Generate boolean mask.
            mask = np.zeros(self.model.dispersion.size, dtype=bool)
            for start, end in value:

                # Allow 'None' to automatically specify edges.
                start = start or -np.inf
                end = end or +np.inf

                censored = (end >= self.model.dispersion) \
                         * (self.model.dispersion >= start)
                mask[censored] = True

            value = mask

        else:
            raise ValueError("cannot interpret censoring mask for label '{}'"\
                .format(label_name))

        dict.__setitem__(self, label_name, value)


    def __getstate__(self):
        """ Return the state of this censoring mask in a serializable form. """

        return self.items()