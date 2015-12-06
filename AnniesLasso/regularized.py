#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A regularized (compressed sensing) version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["RegularizedCannonModel"]

import logging
import numpy as np
import scipy.optimize as op

from . import (cannon, model, utils)

logger = logging.getLogger(__name__)


class RegularizedCannonModel(cannon.CannonModel):
    """
    A L1-regularized edition of The Cannon model for the estimation of arbitrary
    stellar labels.

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

    _descriptive_attributes = ["_label_vector", "_regularization"]
    
    def __init__(self, *args, **kwargs):
        super(RegularizedCannonModel, self).__init__(*args, **kwargs)


    @property
    def regularization(self):
        return self._regularization


    @regularization.setter
    def regularization(self, regularization):
        if regularization is None:
            self._regularization = None
            return None
        
        # Can be positive float, or positive values for all pixels.
        try:
            regularization = float(regularization)
        except (TypeError, ValueError):
            regularization = np.array(regularization).flatten()

            if regularization.size != len(self.dispersion):
                raise ValueError("regularization must be a positive value or "
                                 "an array of positive values for each pixel "
                                 "({0} != {1})".format(
                                    regularization.size,
                                    len(self.dispersion)))

            if any(0 > regularization) \
            or not np.all(np.isfinite(regularization)):
                raise ValueError("regularization terms must be "
                                 "positive and finite")
        else:
            if 0 > regularization or not np.isfinite(regularization):
                raise ValueError("regularization term must be "
                                 "positive and finite")
            regularization = np.ones_like(self.dispersion) * regularization
        self._regularization = regularization
        return None


    # windows to specify zero coefficients for a given label (or terms comprising)
    # that label.


    @model.requires_model_description
    def train(self, **kwargs):
        """
        Train the model based on the training set and the description of the
        label vector, and enforce regularization.
        """
        
        # Initialise the scatter and coefficient arrays.
        N_pixels = len(self.dispersion)
        scatter = np.nan * np.ones(N_pixels)
        label_vector_array = self.label_vector_array
        theta = np.nan * np.ones((N_pixels, label_vector_array.shape[0]))

        # Details for the progressbar.
        pb_kwds = {
            "message": "Training regularized Cannon model from {0} stars with "\
                       "{1} pixels each".format(
                           len(self.training_labels), N_pixels),
            "size": 100 if kwargs.pop("progressbar", True) else -1
        }
        
        if self.pool is None:
            for pixel in utils.progressbar(range(N_pixels), **pb_kwds):
                theta[pixel, :], scatter[pixel] = _fit_pixel(
                    self.training_fluxes[:, pixel], 
                    self.training_flux_uncertainties[:, pixel],
                    label_vector_array, self.regularization[pixel],
                    **kwargs)

        else:
            # Not as nice as just mapping, but necessary for a progress bar.
            process = { pixel: self.pool.apply_async(_fit_pixel, args=(
                    self.training_fluxes[:, pixel], 
                    self.training_flux_uncertainties[:, pixel],
                    label_vector_array, self.regularization[pixel]
                ), kwds=kwargs) \
                for pixel in range(N_pixels) }

            for pixel, proc in utils.progressbar(process.items(), **pb_kwds):
                theta[pixel, :], scatter[pixel] = proc.get()

        self.coefficients, self.scatter = (theta, scatter)
        self._trained = True

        return (theta, scatter)


    def conservative_cross_validation(self, **kwargs):
        """
        Perform conservative cross-validation using cyclic training, validation,
        and test subsets of the labelled data.
        """

        """
        Assign integer probabilities for each star.

        0: for choosing the regularization term.
        1-8 inclusive: training set.
        9: for prediction.
        """

        subset_index = np.random.randint(0, 10, size=len(self.training_labels))

        # Start with an initial value of the regularization.

        raise NotImplementedError



def _fit_pixel(fluxes, flux_uncertainties, label_vector_array, 
    regularization, **kwargs):
    """
    Return the optimal label vector coefficients and scatter for a pixel, given
    the fluxes, uncertainties, and the label vector array.

    :param fluxes:
        The fluxes for the given pixel, from all stars.

    :param flux_uncertainties:
        The 1-sigma flux uncertainties for the given pixel, from all stars.

    :param label_vector_array:
        The label vector array. This should have shape `(N_stars, N_terms + 1)`.

    :param regularization:
        The regularization term.

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """

    _ = kwargs.get("max_uncertainty", 1)
    failed_response = (np.nan * np.ones(label_vector_array.shape[0]), _)
    if np.all(flux_uncertainties >= _):
        return failed_response

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)
    
    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    op_scatter, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        _pixel_scatter_nll, scatter,
        args=(fluxes, flux_uncertainties, label_vector_array, regularization),
        disp=False, full_output=True)

    if warnflag > 0:
        logger.warning("Warning: {}".format([
            "Maximum number of function evaluations made during optimisation.",
            "Maximum number of iterations made during optimisation."
            ][warnflag - 1]))

    # Calculate the coefficients at the optimal scatter value.
    # Note that if we can't solve for the coefficients, we should just set them
    # as zero and send back a giant variance.
    try:
        coefficients, ATCiAinv, variance = cannon._fit_coefficients(
            fluxes, flux_uncertainties, op_scatter, label_vector_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if kwargs.get("debug", False): raise

        return failed_response

    else:
        return (coefficients, op_scatter)


def _pixel_scatter_nll(scatter, fluxes, flux_uncertainties, label_vector_array,
    regularization, **kwargs):
    """
    Return the negative log-likelihood for the scatter in a single pixel.

    :param scatter:
        The model scatter in the pixel.

    :param fluxes:
        The fluxes for a given pixel (in many stars).

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes for a given pixel. This should
        have the same shape as `fluxes`.

    :param label_vector_array:
        The label vector array for each star, for the given pixel.

    :param regularization:
        A regularization term.

    :returns:
        The log-likelihood of the log scatter, given the fluxes and the label
        vector array.

    :raises np.linalg.linalg.LinAlgError:
        If there was an error in inverting a matrix, and `debug` is set to True.
    """

    if 0 > scatter:
        return np.inf

    try:
        # Calculate the coefficients for the given level of scatter.
        theta, ATCiAinv, variance = cannon._fit_coefficients(
            fluxes, flux_uncertainties, scatter, label_vector_array)

    except np.linalg.linalg.LinAlgError:
        if kwargs.get("debug", False): raise
        return np.inf

    model = np.dot(theta, label_vector_array)
    
    return np.sum((fluxes - model)**2 / variance) \
        +  np.sum(np.log(variance)) \
        +  regularization * np.abs(theta).sum()
