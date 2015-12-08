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

    :param training_labels:
        A table with columns as labels, and stars as rows.

    :type training_labels:
        :class:`~astropy.table.Table` or numpy structured array

    :param training_fluxes:
        An array of fluxes for stars in the training set, given as shape
        `(num_stars, num_pixels)`. The `num_stars` should match the number of
        rows in `labels`.

    :type training_fluxes:
        :class:`np.ndarray`

    :param training_flux_uncertainties:
        An array of 1-sigma flux uncertainties for stars in the training set,
        The shape of the `training_flux_uncertainties` should match that of
        `training_fluxes`. 

    :type training_flux_uncertainties:
        :class:`np.ndarray`

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have length `num_pixels`.
    """

    _descriptive_attributes = ["_vectorizer", "_regularization"]
    
    def __init__(self, *args, **kwargs):
        super(RegularizedCannonModel, self).__init__(*args, **kwargs)


    @property
    def regularization(self):
        """
        Return the regularization term for this model.
        """
        return self._regularization


    @regularization.setter
    def regularization(self, regularization):
        """
        Specify the regularization term fot the model, either as a single value
        or a per-pixel value.

        :param regularization:
            The L1-regularization term for the model.
        """
        
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


    @model.requires_model_description
    def train(self, **kwargs):
        """
        Train the model based on the training set and the description of the
        label vector, and enforce regularization.
        """
        
        # Initialise the required arrays.
        N_px = len(self.dispersion)
        design_matrix = self.design_matrix
        scatter = np.nan * np.ones(N_px)
        theta = np.nan * np.ones((N_px, design_matrix.shape[1]))

        pb_kwds = {
            "message": "Training Cannon model from {0} stars with {1} pixels "
                       "each".format(len(self.training_labels), N_px),
            "size": 100 if kwargs.pop("progressbar", True) else -1
        }
        
        if self.pool is None:
            for pixel in utils.progressbar(range(N_px), **pb_kwds):
                theta[pixel, :], scatter[pixel] = _fit_regularized_pixel(
                    self.training_fluxes[:, pixel], 
                    self.training_flux_uncertainties[:, pixel],
                    design_matrix, self.regularization[pixel], **kwargs)

        else:
            # Not as nice as mapping, but necessary if we want a progress bar.
            process = { pixel: self.pool.apply_async(
                    _fit_regularized_pixel,
                    args=(
                        self.training_fluxes[:, pixel], 
                        self.training_flux_uncertainties[:, pixel],
                        design_matrix, self.regularization[pixel]
                    ),
                    kwds=kwargs) \
                for pixel in range(N_px) }

            for pixel, proc in utils.progressbar(process.items(), **pb_kwds):
                theta[pixel, :], scatter[pixel] = proc.get()

        # Save the trained data and finish up.
        self.coefficients = theta
        self.scatter = scatter
        return None


def _fit_regularized_pixel(fluxes, flux_uncertainties, design_matrix,
    regularization, **kwargs):
    """
    Return the optimal label vector coefficients and scatter for a pixel, given
    the fluxes, uncertainties, and the label vector array.

    :param fluxes:
        The fluxes for the given pixel, from all stars.

    :param flux_uncertainties:
        The 1-sigma flux uncertainties for the given pixel, from all stars.

    :param design_matrix:
        The design matrix for the spectral model.

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """

    _ = kwargs.get("max_uncertainty", 1)
    failed_response = (np.nan * np.ones(design_matrix.shape[1]), _)
    if np.all(flux_uncertainties >= _):
        return failed_response

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)
    
    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    op_scatter, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        _model_regularized_pixel_with_scatter, scatter,
        args=(fluxes, flux_uncertainties, design_matrix, regularization),
        disp=False, full_output=True)

    if warnflag > 0:
        logger.warning("Warning: {}".format([
            "Maximum number of function evaluations made during optimisation.",
            "Maximum number of iterations made during optimisation."
            ][warnflag - 1]))

    if not np.isfinite(op_scatter):
        return failed_response

    # If op_scatter is a positive finite value (i.e., if the previous
    # optimisation was successful), this code below *must* work.
    coefficients, ATCiAinv, variance = cannon._fit_theta(
        fluxes, flux_uncertainties, op_scatter, design_matrix)
    return (coefficients, op_scatter)


def _model_regularized_pixel_with_scatter(scatter, fluxes, flux_uncertainties,
    design_matrix, regularization, **kwargs):
    """
    Return the negative log-likelihood for the scatter in a single pixel.

    :param scatter:
        The model scatter in the pixel.

    :param fluxes:
        The fluxes for a given pixel (in many stars).

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes for a given pixel. This should
        have the same shape as `fluxes`.

    :param design_matrix:
        The label vector array for each star, for the given pixel.

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
        theta, ATCiAinv, variance = cannon._fit_theta(
            fluxes, flux_uncertainties, scatter, design_matrix)

    except np.linalg.linalg.LinAlgError:
        if kwargs.get("debug", False): raise
        return np.inf

    model = np.dot(theta, design_matrix.T)
    variance = scatter**2 + flux_uncertainties**2

    return np.sum((fluxes - model)**2 / variance) \
        +  np.sum(np.log(variance)) \
        +  regularization * np.abs(theta[1:]).sum()

