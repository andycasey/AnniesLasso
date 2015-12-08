#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pedestrian version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import logging
import numpy as np
import scipy.optimize as op

from . import (model, utils)

logger = logging.getLogger(__name__)


class CannonModel(model.BaseCannonModel):
    """
    A generalised Cannon model for the estimation of arbitrary stellar labels.

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

    def __init__(self, *args, **kwargs):
        super(CannonModel, self).__init__(*args, **kwargs)


    @model.requires_model_description
    def train(self, **kwargs):
        """
        Train the model based on the training set and the description of the
        label vector.
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
                theta[pixel, :], scatter[pixel] = _fit_pixel(
                    self.training_fluxes[:, pixel], 
                    self.training_flux_uncertainties[:, pixel],
                    design_matrix, **kwargs)

        else:
            # Not as nice as mapping, but necessary if we want a progress bar.
            process = { pixel: self.pool.apply_async(
                    _fit_pixel,
                    args=(
                        self.training_fluxes[:, pixel], 
                        self.training_flux_uncertainties[:, pixel],
                        design_matrix
                    ),
                    kwds=kwargs) \
                for pixel in range(N_px) }

            for pixel, proc in utils.progressbar(process.items(), **pb_kwds):
                theta[pixel, :], scatter[pixel] = proc.get()

        # Save the trained data and finish up.
        self.coefficients = theta
        self.scatter = scatter
        return None
        

    @model.requires_training_wheels
    def predict(self, labels, **kwargs):
        """
        Predict spectra from the trained model, given the labels.

        :param labels:
            The label values to predict model spectra of. The length and order
            should match what is required of the vectorizer
            (`CannonModel.vectorizer.labels`).
        """
        return np.dot(self.coefficients, self.vectorizer(labels).T).T


    @model.requires_training_wheels
    def fit(self, fluxes, flux_uncertainties, **kwargs):
        """
        Solve the labels for given pixel fluxes and uncertainties.

        :param fluxes:
            The normalised fluxes. These should be on the same dispersion scale
            as the trained data.

        :param flux_uncertainties:
            The 1-sigma uncertainties in the fluxes. This should have the same
            shape as `fluxes`.

        :returns:
            The labels.
        """

        fluxes, flux_uncertainties = \
            map(np.atleast_2d, (fluxes, flux_uncertainties))

        N = fluxes.shape[0]
        pb_kwds = {
            "message": "Fitting spectra to {} stars".format(N),
            "size": 100 if kwargs.pop("progressbar", True) and N > 10 else -1
        }
        
        labels = np.nan * np.ones((N, len(self.vectorizer.labels)))
        if self.pool is None:
            for i in utils.progressbar(range(N), **pb_kwds):
                labels[i], _ = _fit_spectrum(
                    self.vectorizer, self.coefficients, self.scatter,
                    fluxes[i], flux_uncertainties[i], **kwargs)

        else:
            processes = { i: self.pool.apply_async(_fit_spectrum,
                    args=(self.vectorizer, self.coefficients, self.scatter,
                        fluxes[i], flux_uncertainties[i]),
                    kwds=kwargs) \
                for i in range(N) }

            for i, process in utils.progressbar(processes.items(), **pb_kwds):
                labels[i], _ = process.get()

        return labels


def _estimate_theta(coefficients, scatter, fluxes, flux_uncertainties, **kwargs):
    """
    Perform a matrix inversion to estimate the vectorizer values given some
    fluxes and associated uncertainties.

    :param fluxes:
        The normalised fluxes. These should be on the same dispersion scale
        as the trained data.

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes. This should have the same
        shape as `fluxes`.

    :returns:
        A two-length tuple containing the label_vector from the matrix
        inversion and the corresponding pixel mask used.
    """

    # Check which pixels to use, then just use those.
    mask = (flux_uncertainties < kwargs.get("max_uncertainty", 1)) \
        * np.isfinite(coefficients[:, 0] * fluxes * flux_uncertainties)

    coefficients = coefficients[mask]
    Cinv = 1.0 / (scatter[mask]**2 + flux_uncertainties[mask]**2)
    A = np.dot(coefficients.T, Cinv[:, None] * coefficients)
    B = np.dot(coefficients.T, Cinv * fluxes[mask])
    return (np.linalg.solve(A, B), mask)


def _fit_spectrum(vectorizer, coefficients, scatter, fluxes, flux_uncertainties,
    **kwargs):
    """
    Solve the labels for given pixel fluxes and uncertainties for a single star.

    :param vectorizer:
        The model vectorizer.

    :param coefficients:
        The trained coefficients for the model.

    :param scatter:
        The trained scatter terms for the model.

    :param fluxes:
        The normalised fluxes. These should be on the same dispersion scale
        as the trained data.

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes. This should have the same
        shape as `fluxes`.

    :returns:
        The labels and covariance matrix.
    """

    # Get an initial estimate of the label vector from a matrix inversion,
    # and then ask the vectorizer to interpret that label vector into the 
    # (approximate) values of the labels that could have produced that 
    # label vector.
    lv, mask = _estimate_theta(
        coefficients, scatter, fluxes, flux_uncertainties)
    initial = vectorizer.get_approximate_labels(lv)

    # Solve for the parameters.
    kwds = {
        "p0": initial,
        "maxfev": 10000,
        "sigma": scatter[mask]**2 + flux_uncertainties[mask]**2,
        "absolute_sigma": True
    }
    kwds.update(kwargs)

    function = lambda c, *l: np.dot(c, vectorizer(l).T).T.flatten()[mask]
    labels, cov = op.curve_fit(function, coefficients, fluxes[mask], **kwds)
    return (labels, cov)


def _fit_pixel(fluxes, flux_uncertainties, design_matrix, **kwargs):
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
        _model_pixel_with_scatter, scatter,
        args=(fluxes, flux_uncertainties, design_matrix),
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
    coefficients, ATCiAinv, variance = _fit_theta(fluxes, flux_uncertainties,
        op_scatter, design_matrix)
    return (coefficients, op_scatter)


def _model_pixel_with_scatter(scatter, fluxes, flux_uncertainties, design_matrix,
    **kwargs):
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
        theta, ATCiAinv, variance = _fit_theta(
            fluxes, flux_uncertainties, scatter, design_matrix)

    except np.linalg.linalg.LinAlgError:
        if kwargs.get("debug", False): raise
        return np.inf

    model = np.dot(theta, design_matrix.T)
    variance = scatter**2 + flux_uncertainties**2
    
    return np.sum((fluxes - model)**2 / variance) + np.sum(np.log(variance))


def _fit_theta(fluxes, flux_uncertainties, scatter, design_matrix):
    """
    Fit model coefficients and scatter to a given set of normalised fluxes for a
    single pixel.

    :param fluxes:
        The normalised fluxes for a single pixel (in many stars).

    :param flux_uncertainties:
        The 1-sigma uncertainties in normalised fluxes. This should have the
        same shape as `fluxes`.

    :param design_matrix:
        The label vector array for each pixel.

    :returns:
        The label vector coefficients for the pixel, the inverse variance matrix
        and the total pixel variance.
    """

    variance = flux_uncertainties**2 + scatter**2
    CiA = design_matrix * np.tile(1./variance, (design_matrix.shape[1], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(design_matrix.T, CiA))

    ATY = np.dot(design_matrix.T, fluxes/variance)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv, variance)