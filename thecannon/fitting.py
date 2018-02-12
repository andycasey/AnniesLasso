#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fitting functions for use in The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["fit_spectrum", "fit_pixel_fixed_scatter", "fit_theta_by_linalg",
    "chi_sq", "L1Norm_variation"]

import logging
import numpy as np
import scipy.optimize as op
from time import time

logger = logging.getLogger(__name__)


def fit_spectrum(flux, ivar, initial_labels, vectorizer, theta, s2, fiducials,
    scales, dispersion=None, **kwargs):
    """
    Fit a single spectrum by least-squared fitting.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variance array for the normalized fluxes.

    :param initial_labels:
        The point(s) to initialize optimization from.

    :param vectorizer:
        The vectorizer to use when fitting the data.

    :param theta:
        The theta coefficients (spectral derivatives) of the trained model.

    :param s2:
        The pixel scatter (s^2) array for each pixel.

    :param dispersion: [optional]
        The dispersion (e.g., wavelength) points for the normalized fluxes.

    :returns:
        A three-length tuple containing: the optimized labels, the covariance
        matrix, and metadata associated with the optimization.
    """

    adjusted_ivar = ivar/(1. + ivar * s2)

    # Exclude non-finite points (e.g., points with zero inverse variance
    # or non-finite flux values, but the latter shouldn't exist anyway).
    use = np.isfinite(flux * adjusted_ivar) * (adjusted_ivar > 0)
    L = len(vectorizer.label_names)

    if not np.any(use):
        logger.warn("No information in spectrum!")
        return (np.nan * np.ones(L), None, {
                "fail_message": "Pixels contained no information"})

    # Splice the arrays we will use most.
    flux = flux[use]
    weights = np.sqrt(adjusted_ivar[use]) # --> 1.0 / sigma
    use_theta = theta[use]

    initial_labels = np.atleast_2d(initial_labels)

    # Check the vectorizer whether it has a derivative built in.
    Dfun = kwargs.pop("Dfun", True)
    if Dfun not in (None, False):
        try:
            vectorizer.get_label_vector_derivative(initial_labels[0])

        except NotImplementedError:
            Dfun = None
            logger.warn("No label vector derivatives available in {}!".format(
                vectorizer))

        except:
            logger.exception("Exception raised when trying to calculate the "\
                             "label vector derivative at the fiducial values:")
            raise

        else:
            # Use the label vector derivative.
            Dfun = lambda parameters: weights * np.dot(use_theta,
                vectorizer.get_label_vector_derivative(parameters)).T

    else:
        Dfun = None

    def func(parameters):
        return np.dot(use_theta, vectorizer(parameters))[:, 0]

    def residuals(parameters):
        return weights * (func(parameters) - flux)

    kwds = {
        "func": residuals,
        "Dfun": Dfun,
        "col_deriv": True,

        # These get passed through to leastsq:
        "ftol": 7./3 - 4./3 - 1, # Machine precision.
        "xtol": 7./3 - 4./3 - 1, # Machine precision.
        "gtol": 0.0,
        "maxfev": 100000, # MAGIC
        "epsfcn": None,
        "factor": 1.0,
    }

    # Only update the keywords with things that op.curve_fit/op.leastsq expects.
    for key in set(kwargs).intersection(kwds):
        kwds[key] = kwargs[key]

    results = []
    for x0 in initial_labels:

        try:
            op_labels, cov, meta, mesg, ier = op.leastsq(
                x0=(x0 - fiducials)/scales, full_output=True, **kwds)

        except RuntimeError:
            logger.exception("Exception in fitting from {}".format(x0))
            continue

        meta.update(
            dict(x0=x0, chi_sq=np.sum(meta["fvec"]**2), ier=ier, mesg=mesg))
        results.append((op_labels, cov, meta))

    if len(results) == 0:
        logger.warn("No results found!")
        return (np.nan * np.ones(L), None, dict(fail_message="No results found"))

    best_result_index = np.nanargmin([m["chi_sq"] for (o, c, m) in results])
    op_labels, cov, meta = results[best_result_index]

    # De-scale the optimized labels.
    meta["model_flux"] = func(op_labels)
    op_labels = op_labels * scales + fiducials

    if np.allclose(op_labels, meta["x0"]):
        logger.warn(
            "Discarding optimized result because it is exactly the same as the "
            "initial value!")

        # We are in dire straits. We should not trust the result.
        op_labels *= np.nan
        meta["fail_message"] = "Optimized result same as initial value."

    if cov is None:
        cov = np.nan * np.ones((len(op_labels), len(op_labels)))

    if not np.any(np.isfinite(cov)):
        logger.warn("Non-finite covariance matrix returned!")

    # Save additional information.
    meta.update({
        "method": "leastsq",
        "label_names": vectorizer.label_names,
        "best_result_index": best_result_index,
        "derivatives_used": Dfun is not None,
        "snr": np.nanmedian(flux * weights),
        "r_chi_sq": meta["chi_sq"]/(use.sum() - L - 1),
    })
    for key in ("ftol", "xtol", "gtol", "maxfev", "factor", "epsfcn"):
        meta[key] = kwds[key]

    return (op_labels, cov, meta)



def fit_theta_by_linalg(flux, ivar, s2, design_matrix):
    """
    Fit theta coefficients to a set of normalized fluxes for a single pixel.

    :param flux:
        The normalized fluxes for a single pixel (across many stars).

    :param ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param s2:
        The noise residual (squared scatter term) to adopt in the pixel.

    :param design_matrix:
        The model design matrix.

    :returns:
        The label vector coefficients for the pixel, and the inverse variance
        matrix.
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

    :param theta:
        The theta coefficients.

    :param design_matrix:
        The model design matrix.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variances of the normalized flux values.

    :param axis: [optional]
        The axis to sum the chi-squared values across.

    :param gradient: [optional]
        Return the chi-squared value and its derivatives (Jacobian).

    :returns:
        The chi-squared difference between the spectral model and flux, and
        optionally, the Jacobian.
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

    :returns:
        A two-length tuple containing: the L1 norm of theta (except the first
        entry), and the derivative of the L1 norm of theta.
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


def fit_pixel_fixed_scatter(flux, ivar, initial_thetas, design_matrix,
    regularization, censoring_mask, **kwargs):
    """
    Fit theta coefficients and noise residual for a single pixel, using
    an initially fixed scatter value.

    :param flux:
        The normalized flux values.

    :param ivar:
        The inverse variance array for the normalized fluxes.

    :param initial_thetas:
        A list of initial theta values to start from, and their source. For
        example: `[(theta_0, "guess"), (theta_1, "old_theta")]

    :param design_matrix:
        The model design matrix.

    :param regularization:
        The regularization strength to apply during optimization (Lambda).

    :param censoring_mask:
        A per-label censoring mask for each pixel.

    :keyword op_method:
        The optimization method to use. Valid options are: `l_bfgs_b`, `powell`.

    :keyword op_kwds:
        A dictionary of arguments that will be provided to the optimizer.

    :returns:
        The optimized theta coefficients, the noise residual `s2`, and
        metadata related to the optimization process.
    """

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

    op_kwds = dict(args=(design_matrix, flux, ivar, regularization),
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
        args = list(op_kwds["args"])
        args.append(False)
        op_kwds["args"] = tuple(args)

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
