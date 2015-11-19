#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pedestrian version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import numpy as np

from . import model


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

    __data_attributes = ["training_labels", "training_fluxes",
        "training_flux_uncertainties"]
    __trained_attributes = ["coefficients", "scatter", "pivot_offsets"]
    __forbidden_label_characters = "^*"

    def __init__(self, labels, fluxes, flux_uncertainties, dispersion=None,
        threads=1, pool=None, live_dangerously=False):

        super(CannonModel, self).__init__(labels, fluxes, flux_uncertainties,
            dispersion=dispersion, threads=threads, pool=pool,
            live_dangerously=live_dangerously)


    def train(self, **kwargs):
        """
        Train the model based on the training set and the description of the
        label vector.
        """
        
        # Initialise the scatter and coefficient arrays.
        N_px = self.number_of_pixels
        scatter = np.nan * np.ones(N_px)
        theta = np.nan * np.ones((N_px, len(self.label_vector)))

        fluxes, flux_uncertainties = (
            self.training_fluxes[~self.training_set_mask], 
            self.training_flux_uncertainties[~self.training_set_mask])

        # Details for the progressbar.
        pb_kwds = {
            "message": "Training Cannon model from {0} stars with {1} pixels "
                       "each".format(self.training_set_size, N_px),
            "size": 100 if kwargs.pop("progressbar", True) else -1
        }
        
        # TODO: Revisit this parallelism.
        if self.pool is None:
            for pixel in utils.progressbar(range(N_px), **pb_kwds):
                theta[pixel, :], scatter[pixel] = _fit_pixel(
                    fluxes[:, pixel], flux_uncertainties[:, pixel],
                    self.label_vector_array, **kwargs)

        else:
            processes = []
            for pixel in range(N_px):
                process = self.pool.apply_async(_fit_pixel,
                    args=(fluxes[:, pixel], flux_uncertainties[:, pixel],
                        self.label_vector_array),
                    kwds=kwargs)
                processes.append((pixel, process))

            for pixel, process in utils.progressbar(processes, **pb_kwds):
                theta[pixel, :], scatter[pixel] = process.get()

        self.theta = theta
        self.scatter = scatter
        self.pivot_offsets = offsets

        self._trained = True

        return (theta, scatter, offsets)




    @model.requires_training
    def predict(self, *args, **kwargs):
        raise NotImplementedError("The predict method must be "
                                  "implemented by subclasses")

    @model.requires_training
    def solve_labels(self, *args, **kwargs):
        raise NotImplementedError("The solve_labels method must be "
                                  "implemented by subclasses")

    @model.requires_training
    def cross_validate(self, *args, **kwargs):
        """
        Perform leave-one-out cross-validation on the training set.
        """

        debug = kwargs.get("debug", False)
        
        N_realisations = self.training_set_size
        N_training_set = self.get_training_set_size(include_masked=True)
        inferred = np.nan * np.ones((N_training_set, self.number_of_parameters))

        for i in range(N_training_set):
            if self.training_set_mask[i]: continue

            mask = self.training_set_mask.copy()
            mask[i] = True

            # Create a clean model to use so we don't overwrite self.
            model = self.__class__(
                self.labels[~mask], self.training_fluxes[~mask],
                self.training_flux_uncertainties[~mask], **kwargs)

            # Initialise the label vector description, etc. This will need
            # all the training_attributes from the current model.
            raise NotImplementedError


            # Solve for the one object left out.
            try:
                inferred[i, :] = model.solve_labels(
                    self.training_fluxes[i, :],
                    self.training_flux_uncertainties[i, :])

            except:
                logger.exception("Exception during cross-validation on object "
                                 "with index {0}:".format(i))
                if debug: raise

        # TODO:
        # Return parameter names as well? the expected labels?
        return inferred[~self.training_set_mask, :]


    @model.requires_training
    def cross_validate_by_label(self, *args, **kwargs):
        raise NotImplementedError("not done yet")


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
        print("Warning: {}".format([
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
        print("Failed to calculate coefficients")
        if kwargs.get("debug", False): raise

        return (np.zeros(label_vector_array.shape[0]), 10e8)

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
