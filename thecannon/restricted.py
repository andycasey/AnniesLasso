#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A restricted Cannon model where bounds are placed on theta coefficients in order
to make the model more physically realistic and limit information propagated
through abundance correlations.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["RestrictedCannonModel"]

import logging
from .model import CannonModel

logger = logging.getLogger(__name__)


class RestrictedCannonModel(CannonModel):
    """
    A model for The Cannon which includes L1 regularization, pixel censoring,
    and is capable of placing bounds on theta coefficients in order to make the
    model more physically realistic and limit information propagated through
    abundance correlations.

    :param training_set_labels:
        A set of objects with labels known to high fidelity. This can be 
        given as a numpy structured array, or an astropy table.

    :param training_set_flux:
        An array of normalised fluxes for stars in the labelled set, given 
        as shape `(num_stars, num_pixels)`. The `num_stars` should match the
        number of rows in `training_set_labels`.

    :param training_set_ivar:
        An array of inverse variances on the normalized fluxes for stars in 
        the training set. The shape of the `training_set_ivar` array should
        match that of `training_set_flux`.

    :param vectorizer:
        A vectorizer to take input labels and produce a design matrix. This
        should be a sub-class of `vectorizer.BaseVectorizer`.

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have a size of `num_pixels`.
    
    :param regularization: [optional]
        The strength of the L1 regularization. This should either be `None`,
        a float-type value for single regularization strength for all pixels,
        or a float-like array of length `num_pixels`.

    :param censors: [optional]
        A dictionary containing label names as keys and boolean censoring
        masks as values.

    :param theta_bounds: [optional]
        A dictionary containing label names as keys and two-length tuples as
        values, indicating acceptable minimum and maximum values. Specify
        `None` to indicate no limit on a boundary.
    """

    def __init__(self, training_set_labels, training_set_flux, training_set_ivar,
        vectorizer, dispersion=None, regularization=None, censors=None, 
        theta_bounds=None, **kwargs):

        super(RestrictedCannonModel, self).__init__(training_set_labels,
            training_set_flux, training_set_ivar, vectorizer, 
            dispersion=dispersion, regularization=regularization, 
            censors=censors, **kwargs)

        self.theta_bounds = theta_bounds
        return None


    @property
    def theta_bounds(self):
        """ Return the boundaries placed on theta coefficients. """
        return self._theta_bounds


    @theta_bounds.setter
    def theta_bounds(self, theta_bounds):
        """
        Set lower and upper boundaries on specific theta coefficients.

        :param theta_bounds:
            A dictionary containing vectorizer terms as keys and two-length 
            tuples as values, indicating acceptable minimum and maximum values. 
            Specify `None` to indicate no limit on a boundary. For example:
            `theta_bounds={"FE_H": (None, 0), "TEFF^3": (None, None)}`
        """
        theta_bounds = {} if theta_bounds is None else theta_bounds
        if isinstance(theta_bounds, dict):
            
            label_vector = self.vectorizer.human_readable_label_vector
            terms = label_vector.split(" + ")
            checked_bounds = {}
            for term in theta_bounds.keys():
                bounds = theta_bounds[term]
                term = str(term)
                
                if term not in terms:
                    logging.warn("Boundary on term '{}' ignored because it is "
                                 "not in the label vector: {}".format(
                                    term, label_vector))
                else:
                    if len(bounds) != 2:
                        raise ValueError("bounds must be a two-length tuple")
                    if None not in bounds and bounds[1] < bounds[0]:
                        raise ValueError("bounds must be in (min, max) order")

                    checked_bounds[term] = bounds

            self._theta_bounds = checked_bounds

        else:
            raise TypeError("theta_bounds must be a dictionary-like object")



    def train(self, threads=None, op_kwds=None):
        """
        Train the model.

        :param threads: [optional]
            The number of parallel threads to use.

        :param op_kwds:
            Keyword arguments to provide directly to the optimization function.

        :returns:
            A three-length tuple containing the spectral coefficients `theta`,
            the squared scatter term at each pixel `s2`, and metadata related to
            the training of each pixel.
        """

        # Generate the optimization bounds based on self.theta_bounds.
        op_bounds = [self.theta_bounds.get(term, (None, None)) \
            for term in self.vectorizer.human_readable_label_vector.split(" + ")]

        kwds = dict(op_method="l_bfgs_b", op_strict=False, op_kwds=(op_kwds or {}))
        kwds["op_kwds"].update(bounds=op_bounds)
        
        return super(RestrictedCannonModel, self).train(threads=threads, **kwds)
