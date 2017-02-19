#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A regularized (compressed sensing) version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import logging
import multiprocessing as mp
import numpy as np
import scipy.optimize as op
from time import time

from .base import BaseCannonModel
from . import utils

logger = logging.getLogger(__name__)


class CannonModel(BaseCannonModel):
    """
    A model for The Cannon which includes L1 regularization and censoring.
    """

    def __init__(self, training_set_labels, training_set_flux, training_set_ivar,
        vectorizer, dispersion=None, regularization=None, censoring=None,
        **kwargs):
        """
        Create a model for The Cannon given a training set and model description.

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

        :param dispersion: [optional]
            The dispersion values corresponding to the given pixels. If provided, 
            this should have length `num_pixels`.
        
        :param regularization: [optional]
            The strength of the L1 regularization. This should either be `None`,
            a float-type value for single regularization strength for all pixels,
            or a float-like array of length `num_pixels`.

        :param censoring: [optional]

        """

        self._training_set_labels = np.array(
            [training_set_labels[ln] for ln in vectorizer.label_names]).T
        self._training_set_flux = np.atleast_2d(training_set_flux)
        self._training_set_ivar = np.atleast_2d(training_set_ivar)
        self._dispersion = dispersion

        # Check that the flux and ivar are valid, and dispersion if given.
        self._verify_training_data()

        # Offset and scale the training set labels.
        self._scales = np.ptp(
            np.percentile(self.training_set_labels, [2.5, 97.5], axis=0), axis=0)
        self._fiducials = np.percentile(self.training_set_labels, 50, axis=0)
        self._scaled_training_set_labels = (self.training_set_labels - self._fiducials)/self._scales

        # Save the vectorizer and create a design matrix.
        self._vectorizer = vectorizer
        self._design_matrix = vectorizer(self._scaled_training_set_labels)

        # Check the regularization and censoring.
        self.regularization = regularization

        self._theta, self._s2 = (None, None)
        return None



    def train(self, threads=None, **kwargs):
        """
        Train the model.

        :param threads: [optional]
            The number of parallel threads to use.

        :returns:
            A three-length tuple containing the spectral coefficients `theta`,
            the squared scatter term at each pixel `s2`, and metadata related to
            the training of each pixel.
        """

        S, P = self.training_set_flux.shape
        T = self.design_matrix.shape[1]

        logger.info("Training {0}-label {1} with {2} stars and {3} pixels/star"\
            .format(len(self.vectorizer.label_names), type(self).__name__, S, P))

        # Parallelise out.
        if threads in (1, None):
            mapper, pool = (map, None)

        else:
            pool = mp.Pool(threads)
            mapper = pool.map

        func = utils.wrapper(_fit_pixel_fixed_scatter, None, {}, P)

        metadata = []
        theta = np.nan * np.ones((P, T))
        s2 = np.nan * np.ones(P)

        for pixel, (flux, ivar) \
        in enumerate(zip(self.training_set_flux.T, self.training_set_ivar.T)):

            args = (
                flux, ivar, 
                self._initial_theta(pixel),
                self.design_matrix,
                self._pixel_access(self.regularization, pixel, 0.0),
                None
            )
            (pixel_theta, pixel_s2, pixel_metadata), = mapper(func, [args])

            metadata.append(pixel_metadata)
            theta[pixel], s2[pixel] = (pixel_theta, pixel_s2)

        self._theta, self._s2, self._training_metadata = (theta, s2, metadata)

        if pool is not None:
            pool.close()
            pool.join()

        return (theta, s2, metadata)



    def test(self, flux, ivar, initial_labels=None, full_output=False, 
        threads=None, **kwargs):


        if threads in (1, None):
            mapper, pool = (map, None)

        else:
            pool = mp.Pool(threads)
            mapper = pool.map


        initial_labels = np.atleast_2d(initial_labels or self._fiducials)

        func = utils.wrapper(
            _fit_spectrum, (initial_labels, self.vectorizer, self.theta, self.s2),
            message="Fitting {} spectra".format(len(flux)))

        args = (np.atleast_2d(flux), np.atleast_2d(ivar))

        output = mapper(func, args)

        raise a


    def fit(self, *args, **kwargs):
        return self.test(*args, **kwargs)



    def _initial_theta(self, pixel_index, **kwargs):
        """
        Return a list of guesses of the spectral coefficients for the given
        pixel index.

        :param pixel_index:
            The zero-indexed integer of the pixel.

        :returns:
            A list of initial theta guesses, and the source of each guess.
        """

        # Preference:
        # - a previously trained theta value
        # - an estimate from linear algebra
        # - a neighbouring pixel's value
        # - a fiducial value

        guesses = []

        if self.theta is not None:
            # Previously trained theta value.
            if np.all(np.isfinite(self.theta[pixel_index])):
                guesses.append((self.theta[pixel_index], "previously_trained"))

        # Estimate from linear algebra.
        theta, cov = _fit_theta_by_linalg(
            self.training_set_flux[:, pixel_index],
            self.training_set_ivar[:, pixel_index],
            s2=kwargs.get("s2", 0.0), design_matrix=self.design_matrix)

        if np.all(np.isfinite(theta)):
            guesses.append((theta, "linear_algebra"))

        if self.theta is not None:
            # Neighbouring pixels value.
            for neighbour_pixel_index in set(np.clip(
                [pixel_index - 1, pixel_index + 1], 
                0, self.training_set_flux.shape[1] - 1)):

                if np.all(np.isfinite(self.theta[neighbour_pixel_index])):
                    guesses.append(
                        (self.theta[neighbour_pixel_index], "neighbour_pixel"))

        # Fiducial value.
        fiducial = np.hstack([1.0, np.zeros(len(self.vectorizer.terms))])
        guesses.append((fiducial, "fiducial"))

        return guesses






def _fit_theta_by_linalg(flux, ivar, s2, design_matrix):
    """
    Fit theta coefficients to a set of normalized fluxes for a single pixel.

    :param normalized_flux:
        The normalized fluxes for a single pixel (across many stars).

    :param normalized_ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param scatter:
        The additional scatter to adopt in the pixel.

    :param design_matrix:
        The model design matrix.

    :returns:
        The label vector coefficients for the pixel, the inverse variance matrix
        and the total inverse variance.
    """

    adjusted_ivar = ivar/(1. + ivar * s2)
    CiA = design_matrix * np.tile(adjusted_ivar, (design_matrix.shape[1], 1)).T
    try:
        ATCiAinv = np.linalg.inv(np.dot(design_matrix.T, CiA))
    except np.linalg.linalg.LinAlgError:
        N = design_matrix.shape[1]
        return (np.hstack([1, np.zeros(N - 1)]), np.inf * np.eye(N))

    ATY = np.dot(design_matrix.T, flux * adjusted_ivar)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv)




# TODO: This logic should probably go somewhere else.

    
def chi_sq(theta, design_matrix, flux, ivar, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and flux.
    """
    residuals = np.dot(theta, design_matrix.T) - flux

    ivar_residuals = ivar * residuals
    f = np.sum(ivar_residuals * residuals, axis=axis)
    if not gradient:
        return f

    g = 2.0 * np.dot(design_matrix.T, ivar_residuals)
    return (f, g)


def L1Norm_variation(theta):
    """
    Return the L1 norm of theta (except the first entry) and its derivative.

    :param theta:
        An array of finite values.
    """

    return (np.sum(np.abs(theta[1:])), np.hstack([0.0, np.sign(theta[1:])]))


def _pixel_objective_function_fixed_scatter(theta, design_matrix, flux, ivar, 
    regularization, gradient=True):
    """
    The objective function for a single regularized pixel with fixed scatter.

    :param theta:
        The spectral coefficients.

    :param normalized_flux:
        The normalized flux values for a single pixel across many stars.

    :param adjusted_ivar:
        The adjusted inverse variance of the normalized flux values for a single 
        pixel across many stars. This adjusted inverse variance array should
        already have the scatter included.

    :param regularization:
        The regularization term to scale the L1 norm of theta with.

    :param design_matrix:
        The design matrix for the model.

    :param gradient: [optional]
        Also return the analytic derivative of the objective function.
    """

    if gradient:
        csq, d_csq = chi_sq(theta, design_matrix, flux, ivar, gradient=True)
        L1, d_L1 = L1Norm_variation(theta)

        f = csq + regularization * L1
        g = d_csq + regularization * d_L1
        
        return (f, g)

    else:
        csq = chi_sq(theta, design_matrix, flux, ivar, gradient=False)
        L1, d_L1 = L1Norm_variation(theta)

        return csq + regularization * L1


def _scatter_objective_function(scatter, residuals_squared, ivar):
    adjusted_ivar = ivar/(1.0 + ivar * scatter**2)
    chi_sq = residuals_squared * adjusted_ivar
    return (np.mean(chi_sq) - 1.0)**2
    

def _fit_pixel_fixed_scatter(flux, ivar, initial_thetas, design_matrix, 
    regularization, censoring_mask, **kwargs): 

    if np.sum(ivar) < 1.0 * ivar.size: # MAGIC
        metadata = dict(message="No pixel information.", op_time=0.0)
        fiducial = np.hstack([1.0, np.zeros(design_matrix.shape[1] - 1)])
        return (fiducial, np.inf, metadata) # MAGIC

    feval = []
    for initial_theta, initial_theta_source in initial_thetas:
        feval.append(_pixel_objective_function_fixed_scatter(
            initial_theta, design_matrix, flux, ivar, regularization, False))

    initial_theta, initial_theta_source = initial_thetas[np.nanargmin(feval)]

    # If the initial_theta is the same size as the censored_mask, but different
    # to the design_matrix, then we need to censor the initial theta.
    if censoring_mask is not None:
        raise NotImplementedError

        if initial_theta.size == censoring_mask.size \
        and initial_theta.size != censoring_mask.sum():
            # Censor the initial theta.
            # Note: the fiducial theta (below) will have the correct size because
            #       the design matrix is already censored.
            initial_theta = initial_theta.copy()[censoring_mask]

    op_kwds = dict(args=[design_matrix, flux, ivar, regularization, True], 
        disp=False, maxfun=np.inf, maxiter=np.inf)

    # Allow either l_bfgs_b or powell
    t_init = time()
    op_method = kwargs.get("op_method", "l_bfgs_b").lower()
    if op_method == "l_bfgs_b":

        op_kwds.update(m=design_matrix.shape[1], factr=10.0, pgtol=1e-6)
        op_kwds.update(kwargs.get("op_kwds", {}))

        op_params, fopt, metadata = op.fmin_l_bfgs_b(
            _pixel_objective_function_fixed_scatter, initial_theta, 
            fprime=None, approx_grad=None, **op_kwds)

        metadata.update(dict(fopt=fopt))

    elif op_method == "powell":
        
        op_kwds.update(xtol=1e-6, ftol=1e-6)
        op_kwds.update(kwargs.get("op_kwds", {}))

        # Set 'False' in args so that we don't return the gradient, because fmin
        # doesn't want it.
        op_kwds["args"][-1] = False

        t_init = time()

        op_params, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
            _pixel_objective_function_fixed_scatter, initial_theta,
            full_output=True, **op_kwds)

        metadata = dict(fopt=fopt, direc=direc, n_iter=n_iter, n_funcs=n_funcs,
            warnflag=warnflag)

    else:
        raise ValueError("unknown optimization method '{}' -- "
                         "powell or l_bfgs_b are available".format(op_method))

    # Additional metadata common to both optimizers.
    metadata.update(dict(op_method=op_method, op_time=time() - t_init,
        initial_theta=initial_theta, initial_theta_source=initial_theta_source))

    # De-censor the optimized parameters.
    if censoring_mask is not None:
        theta = np.zeros(censoring_mask.size)
        theta[censoring_mask] = op_params

    else:
        theta = op_params

    # Fit the scatter.
    residuals_squared = (flux - np.dot(theta, design_matrix.T))**2
    scatter = op.fmin(_scatter_objective_function, 0.0,
        args=(residuals_squared, ivar), disp=False)

    return (theta, scatter**2, metadata)
