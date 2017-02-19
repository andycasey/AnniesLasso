#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities to deal with wavelength censoring.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["create_mask", "censor_design_matrix"]

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _verify_censoring(censoring_dict, num_pixels, label_names):
    """
    Verify that the censoring dictionary provided is valid.

    :param censoring_dict:
        A dictionary containing label names as keys, and boolean masks as values.

    :param num_pixels:
        The number of pixels per spectrum.

    :param label_names:
        The label names used in the vectorizer.

    :returns:
        A valid censoring dictionary.
    """

    if censoring_dict is None:
        return {}

    if not isinstance(censoring_dict, dict):
        raise TypeError("censors must be provided as a dictionary")

    valid_censoring_dict = censoring_dict.copy()

    unknown_label_names = list(set(censoring_dict).difference(label_names))
    if len(unknown_label_names) > 0:
        logger.warn("Unkown label names provided in censoring dictionary. "
                    "These censors will be ignored: {}".format(
                        ", ".join(unknown_label_names)))

        for unknown_label_name in unknown_label_names:
            del valid_censoring_dict[unknown_label_name]

    for label_name in valid_censoring_dict.keys():
        mask = np.array(valid_censoring_dict[label_name]).flatten().astype(bool)
        if mask.size != num_pixels:
            raise ValueError("wrong shape for '{}' censoring mask ({} != {})"\
                .format(label_name, mask.size, num_pixels))

        valid_censoring_dict[label_name] = mask

    return valid_censoring_dict


def create_mask(dispersion, censored_regions):
    """
    Return a boolean censoring mask based on a structured list of (start, end)
    regions.
    """

    mask = np.zeros(dispersion.size, dtype=bool)

    if isinstance(censored_regions[0], (int, float)):
        censored_regions = [censored_regions]

    for start, end in censored_regions:
        start, end = (start or -np.inf, end or +np.inf)

        censored = (end >= dispersion) * (dispersion >= start)
        mask[censored] = True

    return mask




def design_matrix_mask(censors, vectorizer, num_pixels):
    """
    Return a mask of which indices in the design matrix columns should be
    used for a given pixel. 
    """        

    # Parse all the terms once-off.
    mapper = {}
    pixel_masks = np.atleast_2d(list(map(list, censors.values())))
    for i, terms in enumerate(vectorizer.terms):
        for label_index, power in terms:
            # Let's map this directly to the censors that we actually have.
            try:
                censor_index = list(censors.keys()).index(
                    vectorizer.label_names[label_index])

            except ValueError:
                # Label name is not censored, so we don't care.
                continue

            else:
                # Initialize a list if necessary.
                mapper.setdefault(censor_index, [])

                # Note that we add +1 because the first term in the design
                # matrix columns will actually be the pivot point.
                mapper[censor_index].append(1 + i)

    # We already know the number of terms from i.
    raise CHECKTHIS_AND_WHAT_IF_NO_GOOD_CENSORS_DICT
    mask = np.ones((pixel_masks.shape[0], 2 + i), dtype=bool)
    for censor_index, pixel in zip(*np.where(pixel_masks)):
        mask[pixel, mapper[censor_index]] = False

    return mask