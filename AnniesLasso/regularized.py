#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A regularized (compressed sensing) version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["L1RegularizedCannonModel"]

import logging
import numpy as np
import scipy.optimize as op

from . import (cannon, utils)

logger = logging.getLogger(__name__)


class L1RegularizedCannonModel(cannon.CannonModel):
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
        super(L1RegularizedCannonModel, self).__init__(*args, **kwargs)


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
        
        regularization = np.array(regularization).flatten()
        if regularization.size == 1:
            regularization = np.ones_like(self.dispersion) * regularization[0]

        elif regularization.size != len(self.dispersion):
            raise ValueError("regularization must be a positive value or "
                             "an array of positive values for each pixel "
                             "({0} != {1})".format(regularization.size,
                                len(self.dispersion)))

        if any(0 > regularization) \
        or not np.all(np.isfinite(regularization)):
            raise ValueError("regularization terms must be "
                             "positive and finite")
        self._regularization = regularization
        return None


    def train(self, fixed_scatter=True, **kwargs):
        """
        Train the model based on the labelled set using the given vectorizer.

        :param fixed_scatter: [optional]
            Fix the scatter terms and do not solve for them during the training
            phase. If set to `True`, the `s2` attribute must be already set.
        """

        kwds = {
            "fixed_scatter": fixed_scatter,
            "function": _fit_regularized_pixel,
            "additional_args": [self.regularization, ]
        }
        kwds.update(kwargs)
        super(L1RegularizedCannonModel, self).train(**kwds)


def chi_sq(theta, design_matrix, data, ivar, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and data.
    """
    residuals = np.dot(theta, design_matrix.T) - data

    f = np.sum(ivar * residuals**2, axis=axis)
    if not gradient:
        return f

    g = 2.0 * np.dot(design_matrix.T, ivar * residuals)
    return (f, g)

    
def L1Norm(Q):
    """
    Return the L1 normalization of Q and its derivative.

    :param Q:
        An array of finite values.
    """
    return (np.sum(np.abs(Q)), np.sign(Q))


def _objective_function_for_a_regularized_pixel_with_fixed_scatter(theta, 
    normalized_flux, adjusted_ivar, regularization, design_matrix,
    gradient=True):
    """
    The objective function for a single regularized pixel with fixed scatter.

    :param theta:
        The theta parameters to solve for.

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

    csq, d_csq = chi_sq(theta, design_matrix, normalized_flux, adjusted_ivar)
    L1, d_L1 = L1Norm(theta)

    # We are using a variation of L1 norm that ignores the first coefficient.
    L1 -= np.abs(theta[0])

    f = csq + regularization * L1
    if not gradient:
        return f

    g = d_csq + regularization * d_L1
    #print(f.sum(),g.sum())
    return (f, g)


def _fit_regularized_pixel(initial_theta, initial_s2, normalized_flux, 
    normalized_ivar, regularization, design_matrix, fixed_scatter, **kwargs):

    design_matrix = utils._unpack_value(design_matrix)

    # Any actual information in these pixels?
    if np.sum(normalized_ivar) < 1. * normalized_ivar.size: # MAGIC 
        fiducial_theta = np.hstack([1, np.zeros(design_matrix.shape[1] - 1)])
        metadata = { "message": "No pixel information." }
        return (np.hstack([fiducial_theta, np.inf]), metadata)

    # Set up the method and arguments.
    if fixed_scatter:
        func = _objective_function_for_a_regularized_pixel_with_fixed_scatter
        adjusted_ivar = normalized_ivar/(1. + normalized_ivar * initial_s2)
        args = (normalized_flux, adjusted_ivar, regularization, design_matrix)

    else:
        raise WTF
        #p0 = np.hstack([initial_theta, scatter])
        #func = _fit_pixel_with_fixed_regularization
        #args = (normalized_flux, normalized_ivar, regularization, design_matrix)

    # Set up the initial theta value.
    if initial_theta is None:
        initial_theta, _, __ = cannon._fit_theta(
            normalized_flux, normalized_ivar, initial_s2, design_matrix)

    # Is the fiducial theta a better starting point?
    fiducial_theta = np.hstack([1, np.zeros(design_matrix.shape[1] - 1)])
    if func(fiducial_theta, *args)[0] < func(initial_theta, *args)[0]:
        initial_theta = fiducial_theta

    # Starting point for optimization.
    p0 = np.array(initial_theta)    if fixed_scatter \
                                    else np.hstack([initial_theta, initial_s2])

    # Prepare keywords for optimization.
    kwds = {
        "args": args,
        "disp": False,
        "maxfun": np.inf,
        "maxiter": np.inf,
    }

    # Keywords specific to BFGS (and default values).
    bfgs_terms = {
        "m": p0.size,
        "factr": 10.0,
        "pgtol": 1e-6,
    }
    bfgs_terms.update(kwargs.pop("op_bfgs_kwargs", {}))
    kwds.update(bfgs_terms)

    logger.debug("BFGS keywords: {}".format(kwds))

    op_params, fopt, d = op.fmin_l_bfgs_b(
        func, p0, fprime=None, approx_grad=False, **kwds)

    metadata = {
        "bfgs_fopt": fopt,
        "bfgs_dict": d
    }

    if d["warnflag"] > 0:
        
        # Run Powell's method instead.
        # Default values:
        kwds.update({
            "xtol": 1e-6,
            "ftol": 1e-6
        })
        kwds.update(kwargs.get("op_fmin_kwargs", {}))
        for k in bfgs_terms:
            del kwds[k]

        # Add 'False' to args so that we don't return gradient because fmin does
        # not want it.
        kwds["args"] = tuple(list(kwds["args"]) + [False])        

        logger.debug("fmin_powell keywords: {}".format(kwds))
        op_params, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
            func, op_params, full_output=True, **kwds)

        if warnflag > 0:
            logger.warning("""
            BFGS stopped prematurely:
                {0}
            And then Powell optimization failed:
                {1}
            """.format(d["task"], [
                    "MAXIMUM NUMBER OF FUNCTION EVALUATIONS.",
                    "MAXIMUM NUMBER OF ITERATIONS."
                ][warnflag - 1]))
        
        metadata.update({
            "fmin_fopt": fopt,
            "fmin_niter": n_iter,
            "fmin_nfuncs": n_funcs,
            "fmin_warnflag": warnflag
        })
        
    result = np.hstack([op_params, initial_s2]) if fixed_scatter else op_params

    return (result, metadata)



