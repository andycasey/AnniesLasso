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
from sys import stdout

from . import (cannon, model, utils)

logger = logging.getLogger(__name__)


class RegularizedCannonModel(cannon.CannonModel):
    """
    A L1-regularized edition of The Cannon model for the estimation of arbitrary
    stellar labels.

    :param labelled_set:
        A set of labelled objects. The most common input form is a table with
        columns as labels, and stars/objects as rows.

    :type labelled_set:
        :class:`~astropy.table.Table`, numpy structured array

    :param normalized_flux:
        An array of normalized fluxes for stars in the labelled set, given as
        shape `(num_stars, num_pixels)`. The `num_stars` should match the number
        of rows in `labelled_set`.

    :type normalized_flux:
        :class:`np.ndarray`

    :param normalized_ivar:
        An array of inverse variances on the normalized fluxes for stars in the
        labelled set. The shape of the `normalized_ivar` array should match that
        of `normalized_flux`.

    :type normalized_ivar:
        :class:`np.ndarray`

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have length `num_pixels`.

    :param threads: [optional]
        Specify the number of parallel threads to use. If `threads > 1`, the
        training and prediction phases will be automagically parallelised.

    :param pool: [optional]
        Specify an optional multiprocessing pool to map jobs onto.
        This argument is only used if specified and if `threads > 1`.
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
                                 "({0} != {1})".format(regularization.size,
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
    def train(self, fixed_scatter=False, **kwargs):
        """
        Train the model based on the labelled set using the given vectorizer and
        regularization terms.

        :param fix_scatter: [optional]
            Fix the scatter terms and do not solve for them during the training
            phase. If set to `True`, the `s2` attribute must be already set.
        """
        
        # Initialise the required arrays.
        N_px = len(self.dispersion)
        design_matrix = self.design_matrix
        
        scatter = np.nan * np.ones(N_px)
        theta = np.nan * np.ones((N_px, design_matrix.shape[1]))

        pb_kwds = {
            "message": "Training L1-regularized Cannon model from {0} stars "
                       "with {1} pixels and a {2:.0e} mean regularization "
                       "factor".format(len(self.labelled_set), N_px,
                            np.mean(self.regularization)),
            "size": 100 if kwargs.pop("progressbar", True) else -1
        }
        
        if self.pool is None:
            for pixel in utils.progressbar(range(N_px), **pb_kwds):
                theta[pixel, :], scatter[pixel] = _fit_pixel(
                    self.normalized_flux[:, pixel], 
                    self.normalized_ivar[:, pixel],
                    design_matrix, self.regularization[pixel], **kwargs)

        else:
            # Not as nice as mapping, but necessary if we want a progress bar.
            process = { pixel: self.pool.apply_async(
                    _fit_pixel,
                    args=(
                        self.normalized_flux[:, pixel], 
                        self.normalized_ivar[:, pixel],
                        design_matrix,
                        self.regularization[pixel],
                    ),
                    kwds=kwargs) \
                for pixel in range(N_px) }

            for pixel, proc in utils.progressbar(process.items(), **pb_kwds):
                theta[pixel, :], scatter[pixel] = proc.get()

        # Save the trained data and finish up.
        self.theta, self.s2 = theta, scatter**2
        return None


def L1Norm(Q):
    """
    Return the L1 normalization of Q.

    :param Q:
        An array of finite values.
    """
    return np.sum(np.abs(Q))


def _fit_pixel_with_fixed_regularization(parameters, normalized_flux,
    normalized_ivar, design_matrix, regularization, **kwargs):
    """
    Fit the normalized flux for a single pixel (across many stars) given the
    parameters (scatter, theta) and a fixed regularization term.

    :param parameters:
        The parameters `(scatter, *theta)` to employ.

    :param normalized_flux:
        The normalized flux values for a single pixel across many stars.

    :param normalized_ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param design_matrix:
        The design matrix for the model.

    :param regularization:
        The regularization term to scale the L1 norm of theta with.
    """
    scatter, theta = parameters[0], parameters[1:]

    residuals = np.dot(theta, design_matrix.T) - normalized_flux
    inv_var = normalized_ivar/(1. + normalized_ivar * scatter**2)
    return np.sum(inv_var * residuals**2) - np.sum(np.log(inv_var)) \
        + regularization * L1Norm(theta[1:])

def _fit_pixel_with_fixed_regularization_and_fixed_scatter(theta, scatter,
    normalized_flux, normalized_ivar, design_matrix, regularization, **kwargs):
    """
    Fit the normalized flux for a single pixel (across many stars) given the
    theta parameters, a fixed scatter, and a fixed regularization term.

    :param theta:
        The theta parameters to solve for.

    :param scatter:
        The fixed scatter term to apply.

    :param normalized_flux:
        The normalized flux values for a single pixel across many stars.

    :param normalized_ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param design_matrix:
        The design matrix for the model.

    :param regularization:
        The regularization term to scale the L1 norm of theta with.
    """

    residuals = np.dot(theta, design_matrix.T) - normalized_flux
    inv_var = normalized_ivar/(1. + normalized_ivar * scatter**2)
    return np.sum(inv_var * residuals**2) - np.sum(np.log(inv_var)) \
        + regularization * L1Norm(theta[1:])



def _fit_pixel(normalized_flux, normalized_ivar, design_matrix, regularization,
    scatter=None, **kwargs):
    """
    Return the optimal vectorizer coefficients and variance term for a pixel
    given the normalized flux, the normalized inverse variance, and the design
    matrix.

    :param normalized_flux:
        The normalized flux values for a given pixel, from all stars.

    :param normalized_ivar:
        The inverse variance of the normalized flux values for a given pixel,
        from all stars.

    :param design_matrix:
        The design matrix for the spectral model.

    :param regularization:
        The regularization term for the given pixel.

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """
    kwds = {
        "disp": False,
        "maxiter": np.inf,
        "maxfun": np.inf,
        "full_output": True,
        "retall": False,
    }

    # TODO: allow initial theta to be given as a kwarg
    fix_scatter = scatter is not None
    if scatter is None:
        initial_scatter = kwargs.get("initial_scatter", 0.01)
        initial_theta, _, __ = cannon._fit_theta(normalized_flux, normalized_ivar, 
            initial_scatter, design_matrix)

        # Build the initial guess.
        p0 = np.hstack([initial_scatter, initial_theta])
        func = _fit_pixel_with_fixed_regularization
        kwds["args"] = (normalized_flux, normalized_ivar, design_matrix, regularization)
    

    else:
        func = _fit_pixel_with_fixed_regularization_and_fixed_scatter
        initial_theta = cannon._fit_theta(normalized_flux, normalized_ivar, 
            scatter, design_matrix)
        p0 = initial_theta
        kwds["args"] = (scatter, normlised_flux, normalized_ivar, design_matrix, regularization)


    op_parameters, fopt, direc, n_iter, n_funcalls, warnflag = op.fmin_powell(
        func, p0, **kwds)

    if warnflag > 0:
        stdout.write("\r\n")
        stdout.flush()
        logger.warning("Optimization stopped prematurely: {}".format([
            "Maximum number of function evaluations.",
            "Maximum number of iterations."
            ][warnflag - 1]))

    if fix_scatter:
        theta = op_parameters
    else:
        scatter, theta = op_parameters[0], op_parameters[1:]

    return (theta, scatter)