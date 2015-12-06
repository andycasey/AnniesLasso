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

    _descriptive_attributes = ["_label_vector"]
    _trained_attributes = ["_coefficients", "_scatter", "_pivots"]
    _data_attributes = \
        ["training_labels", "training_fluxes", "training_flux_uncertainties"]
    
    def __init__(self, *args, **kwargs):
        super(CannonModel, self).__init__(*args, **kwargs)


    @model.requires_model_description
    def train(self, **kwargs):
        """
        Train the model based on the training set and the description of the
        label vector.
        """
        
        # Initialise the scatter and coefficient arrays.
        N_px = len(self.dispersion)
        scatter = np.nan * np.ones(N_px)
        label_vector_array = self.label_vector_array
        theta = np.nan * np.ones((N_px, label_vector_array.shape[0]))

        # Details for the progressbar.
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
                    label_vector_array, **kwargs)

        else:
            # Not as nice as just mapping, but necessary for a progress bar.
            process = { pixel: self.pool.apply_async(
                    _fit_pixel,
                    args=(
                        self.training_fluxes[:, pixel], 
                        self.training_flux_uncertainties[:, pixel],
                        label_vector_array
                    ),
                    kwds=kwargs) \
                for pixel in range(N_px) }

            for pixel, proc in utils.progressbar(process.items(), **pb_kwds):
                theta[pixel, :], scatter[pixel] = proc.get()

        # Save the trained state.
        self.coefficients, self.scatter = (theta, scatter)
        self._trained = True

        return (theta, scatter)


    @model.requires_training_wheels
    def predict(self, labels=None, **labels_as_kwargs):
        """
        Predict spectra from the trained model, given the labels.

        :param labels: [optional]
            The labels required for the trained model. This should be a N-length
            list matching the number of unique terms in the model, in the order
            given by `self.labels`. Alternatively, labels can be explicitly 
            given as keyword arguments.
        """

        labels = self._format_input_labels(labels, **labels_as_kwargs)

        return np.dot(self.coefficients, model._build_label_vector_rows(
            self.label_vector, labels, self.pivots)).flatten()


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
            The labels and covariance matrix.
        """

        label_indices = self._get_lowest_order_label_indices()
        fluxes, flux_uncertainties = map(np.array, (fluxes, flux_uncertainties))

        # TODO: Consider parallelising this, which would mean factoring
        # _fit out of the model class, which gets messy.
        # Since solving for labels is not a big bottleneck (yet), let's leave
        # this.

        full_output = kwargs.pop("full_output", False)
        if fluxes.ndim == 1:
            labels, covariance = \
                self._fit(fluxes, flux_uncertainties, label_indices, **kwargs)
        else:
            N_stars, N_labels = (fluxes.shape[0], len(self.labels))
            labels = np.empty((N_stars, N_labels), dtype=float)
            covariance = np.empty((N_stars, N_labels, N_labels), dtype=float)

            for i, (f, u) in enumerate(zip(fluxes, flux_uncertainties)):
                labels[i, :], covariance[i, :] = \
                    self._fit(f, u, label_indices, **kwargs)

        if full_output:
            return (labels, covariance)
        return labels


    def _fit(self, fluxes, flux_uncertainties, label_indices, **kwargs):
        """
        Solve the labels for given pixel fluxes and uncertainties
        for a single star.

        :param fluxes:
            The normalised fluxes. These should be on the same dispersion scale
            as the trained data.

        :param flux_uncertainties:
            The 1-sigma uncertainties in the fluxes. This should have the same
            shape as `fluxes`.

        :returns:
            The labels and covariance matrix.
        """

        # Check which pixels to use, then just use those.
        use = (flux_uncertainties < kwargs.get("max_uncertainty", 1)) \
            * np.isfinite(self.coefficients[:, 0] * fluxes * flux_uncertainties)

        fluxes = fluxes.copy()[use]
        flux_uncertainties = flux_uncertainties.copy()[use]
        scatter, coefficients = self.scatter[use], self.coefficients[use]

        Cinv = 1.0 / (scatter**2 + flux_uncertainties**2)
        A = np.dot(coefficients.T, Cinv[:, None] * coefficients)
        B = np.dot(coefficients.T, Cinv * fluxes)
        theta_p0 = np.linalg.solve(A, B)

        # Need to match the initial theta coefficients back to label values.
        # (Maybe this should use some general non-linear simultaneous solver?)
        initial = {}
        for index in label_indices:
            if index is None: continue
            label, order = self.label_vector[index][0]
            # The +1 index offset is because the first theta is a scaling.
            value = abs(theta_p0[1 + index])**(1./order)
            if not np.isfinite(value): continue
            initial[label] = value

        # There could be some coefficients that are only used in cross-terms.
        # We could solve for them, or just take them as zero (i.e., near the
        # pivot point of the data set).
        missing = set(self.labels).difference(initial)
        initial.update({ label: 0.0 for label in missing })

        # Create and test the generating function.
        def function(coeffs, *labels):
            return np.dot(coeffs, 
                model._build_label_vector_rows(self.label_vector, 
                    { label: [v] for label, v in zip(self.labels, labels) }
                )).flatten()

        # Solve for the parameters.
        kwds = {
            "p0": np.array([initial[label] for label in self.labels]),
            "maxfev": 10000,
            "sigma": 1.0/np.sqrt(Cinv),
            "absolute_sigma": True
        }
        kwds.update(kwargs)
        labels_opt, cov = op.curve_fit(function, coefficients, fluxes, **kwds)

        # Apply any necessary pivots to put these back to real space.   
        labels_opt += np.array([self.pivots[l] for l in self.labels])
        
        return (labels_opt, cov)


def _fit_pixel(fluxes, flux_uncertainties, label_vector_array, **kwargs):
    """
    Return the optimal label vector coefficients and scatter for a pixel, given
    the fluxes, uncertainties, and the label vector array.

    :param fluxes:
        The fluxes for the given pixel, from all stars.

    :param flux_uncertainties:
        The 1-sigma flux uncertainties for the given pixel, from all stars.

    :param label_vector_array:
        The label vector array. This should have shape `(N_stars, N_terms + 1)`.

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """

    _ = kwargs.get("max_uncertainty", 1)
    failed_response = (np.nan * np.ones(label_vector_array.shape[0]), _)
    if np.all(flux_uncertainties > _):
        return failed_response

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)
    
    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    op_scatter, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        _pixel_scatter_nll, scatter,
        args=(fluxes, flux_uncertainties, label_vector_array),
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
        coefficients, ATCiAinv, variance = _fit_coefficients(
            fluxes, flux_uncertainties, op_scatter, label_vector_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if kwargs.get("debug", False): raise

        return failed_response

    else:
        return (coefficients, op_scatter)


def _pixel_scatter_nll(scatter, fluxes, flux_uncertainties, label_vector_array,
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

    :param label_vector_array:
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
        theta, ATCiAinv, variance = _fit_coefficients(
            fluxes, flux_uncertainties, scatter, label_vector_array)

    except np.linalg.linalg.LinAlgError:
        if kwargs.get("debug", False): raise
        return np.inf

    model = np.dot(theta, label_vector_array)

    return 0.5 * np.sum((fluxes - model)**2 / variance) \
        +  0.5 * np.sum(np.log(variance))


def _fit_coefficients(fluxes, flux_uncertainties, scatter, label_vector_array):
    """
    Fit model coefficients and scatter to a given set of normalised fluxes for a
    single pixel.

    :param fluxes:
        The normalised fluxes for a single pixel (in many stars).

    :param flux_uncertainties:
        The 1-sigma uncertainties in normalised fluxes. This should have the
        same shape as `fluxes`.

    :param label_vector_array:
        The label vector array for each pixel.

    :returns:
        The label vector coefficients for the pixel, the inverse variance matrix
        and the total pixel variance.
    """

    variance = flux_uncertainties**2 + scatter**2
    CiA = label_vector_array.T * \
        np.tile(1./variance, (label_vector_array.shape[0], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(label_vector_array, CiA))

    ATY = np.dot(label_vector_array, fluxes/variance)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv, variance)
