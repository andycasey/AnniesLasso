#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A pedestrian version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import cPickle as pickle
import logging
import numpy as np
import scipy.optimize as op
import tempfile
from six import string_types

from . import (model, utils)

logger = logging.getLogger(__name__)


class CannonModel(model.BaseCannonModel):
    """
    A generalised Cannon model for the estimation of arbitrary stellar labels.

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
    def __init__(self, *args, **kwargs):
        super(CannonModel, self).__init__(*args, **kwargs)


    @model.requires_model_description
    def train(self, fixed_scatter=False, progressbar=True, initial_theta=None,
        use_neighbouring_pixel_theta=False,
        **kwargs):
        """
        Train the model based on the labelled set using the given vectorizer.

        :param fixed_scatter: [optional]
            Fix the scatter terms and do not solve for them during the training
            phase. If set to `True`, the `s2` attribute must be already set.

        :param progressbar: [optional]
            Show a progress bar.
        """
        
        print("OVERWRITING S2 AND FIXED SCATTER")
        self.s2 = 0.0
        self.fixed_scatter = True

        if fixed_scatter and self.s2 is None:
            raise ValueError("intrinsic pixel variance (s2) must be set "
                             "before training if fixed_scatter is set to True")

        # Initialize the scatter.
        p0_scatter = np.sqrt(self.s2) if fixed_scatter \
            else 0.01 * np.ones_like(self.dispersion)

        # Prepare details about any progressbar to show.
        M, N = self.normalized_flux.shape
        message = None if not progressbar else \
            "Training {0}-label {1} with {2} stars and {3} pixels/star".format(
                len(self.vectorizer.label_names), type(self).__name__, M, N)

        # Prepare the method and arguments.
        fitter = kwargs.pop("function", _fit_pixel)
        kwds = {
            "fixed_scatter": fixed_scatter,
            "op_kwargs": kwargs.pop("op_kwargs", {}),
            "op_bfgs_kwargs": kwargs.pop("op_bfgs_kwargs", {})
        }
        #kwds.update(kwargs)

        temporary_filenames = []

        args = [self.normalized_flux.T, self.normalized_ivar.T, p0_scatter]
        args.extend(kwargs.get("additional_args", []))

        if self.pool is None:
            mapper = map
            
            kwds["design_matrix"] = self.design_matrix
            """
            results = []
            previous_theta = [None]
            for row in utils.progressbar(zip(*args), message=message):
                logger.info("passing initial_theta: {}".format(previous_theta[-1]))
                row = list(row)
                row[-1] = previous_theta[-1]
                row = tuple(row)
                results.append(fitter(*row, **kwds))
                previous_theta.append(results[-1][:-1].copy())
            """
            results = []
            previous_theta = [None]
            for j, row in enumerate(utils.progressbar(zip(*args), message=message)):
                if j > 0 and use_neighbouring_pixel_theta:
                    row = list(row)
                    row[-1] = previous_theta[-1]
                    row = tuple(row)
                #row = list(row)
                #raise a
                #row[-1] = initial_theta
                #row = tuple(row)
                #print("ACTUALLY SENDING {}".format(initial_theta))
                results.append(fitter(*row, **kwds))
                if use_neighbouring_pixel_theta:
                    previous_theta[-1] = results[-1][:-1]

            results = np.array(results)

        else:
            mapper = self.pool.map
            
            # Write the design matrix to a temporary file.
            _, temporary_filename = tempfile.mkstemp()
            with open(temporary_filename, "wb") as fp:
                pickle.dump(self.design_matrix, fp, -1)
            kwds["design_matrix"] = temporary_filename
            temporary_filenames.append(temporary_filename)

        # Wrap the function so we can parallelize it out.
        f = utils.wrapper(fitter, None, kwds, N, message=message)
        results = np.array(mapper(f, [row for row in zip(*args)]))

        # Calculate chunk size, etc.
        """
        chunks = kwargs.pop("chunks", 10)
        chunk_size = int(np.ceil(len(self.dispersion)/float(chunks)))
        theta = np.zeros((len(self.dispersion), 1 + len(self.vectorizer.terms)))
        scatter = np.zeros((len(self.dispersion)))
        for i in range(chunks):
            # Time for work.
            logger.info("Chunks: {0} w/ chunk size: {1}".format(chunks,
                chunk_size))
            a, b = (i * chunk_size, (i + 1) * chunk_size)

            sm_nf = sharedmem.empty_like(self.normalized_flux.T[a:b])
            sm_ni = sharedmem.empty_like(self.normalized_ivar.T[a:b])
            sm_p0 = sharedmem.empty_like(p0_scatter[a:b])

            sm_nf[:] = self.normalized_flux.T[a:b]
            sm_ni[:] = self.normalized_ivar.T[a:b]
            sm_p0[:] = p0_scatter[a:b]
            args = [sm_nf, sm_ni, sm_p0]
            args.extend(kwargs.get("additional_args", []))
            
            #args = [self.normalized_flux.T[a:b], self.normalized_ivar.T[a:b], p0_scatter[a:b]]
            #args.extend(kwargs.get("additional_args", []))
        
            #_ = args[a:b]
            
            results = np.array(results)
            
            # Save these:
            logger.info("Saving to temp.pkl and re-chunking..")
            theta[a:b] = results[a:b, :-1]
            scatter[a:b] = results[a:b, -1]

            with open("temp.pkl", "wb") as fp:
                pickle.dump((theta, scatter), fp, -1)
        """

            

        #self.theta = theta
        #self.s2 = scatter**2

        # Clean up any temporary files.
        for filename in temporary_filenames:
            os.remove(filename)

        # Unpack the results.
        self.theta, self.s2 = (results[:, :-1], results[:, -1]**2)
        assert np.all(self.s2 == 0.0)
        return None


    @model.requires_training_wheels
    def predict(self, labels, **kwargs):
        """
        Predict spectra from the trained model, given the labels.

        :param labels:
            The label values to predict model spectra of. The length and order
            should match what is required of the vectorizer
            (`CannonModel.vectorizer.label_names`).
        """
        return np.dot(self.theta, self.vectorizer(labels).T).T


    @model.requires_training_wheels
    def fit(self, normalized_flux, normalized_ivar, **kwargs):
        """
        Solve the labels for the given normalized fluxes and inverse variances.

        :param normalized_flux:
            The normalized fluxes. These should be on the same dispersion scale
            as the trained data.

        :param normalized_ivar:
            The inverse variances of the normalized flux values. This should
            have the same shape as `normalized_flux`.

        :returns:
            The labels.
        """
        normalized_flux = np.atleast_2d(normalized_flux)
        normalized_ivar = np.atleast_2d(normalized_ivar)

        # Prepare the wrapper function and data.
        N_spectra = normalized_flux.shape[0]
        message = None if not kwargs.pop("progressbar", True) \
            else "Fitting {0} spectra".format(N_spectra)
        kwds = {
            "vectorizer": self.vectorizer,
            "theta": self.theta,
            "s2": self.s2
        }
        args = [normalized_flux, normalized_ivar]
        
        f = utils.wrapper(_fit_spectrum, None, kwds, N_spectra, message=message)

        # Do the grunt work.
        mapper = map if self.pool is None else self.pool.map
        labels, cov = map(np.array, zip(*mapper(f, [r for r in zip(*args)])))

        return (labels, cov) if kwargs.get("full_output", False) else labels


def _estimate_label_vector(theta, s2, normalized_flux, normalized_ivar,
    **kwargs):
    """
    Perform a matrix inversion to estimate the values of the label vector given
    some normalized fluxes and associated inverse variances.

    :param theta:
        The theta coefficients obtained from the training phase.

    :param s2:
        The intrinsic pixel variance.

    :param normalized_flux:
        The normalized flux values. These should be on the same dispersion scale
        as the labelled data set.

    :param normalized_ivar:
        The inverse variance of the normalized flux values. This should have the
        same shape as `normalized_flux`.
    """

    inv_var = normalized_ivar/(1. + normalized_ivar * s2)
    A = np.dot(theta.T, inv_var[:, None] * theta)
    B = np.dot(theta.T, inv_var * normalized_flux)
    return np.linalg.solve(A, B)


def _fit_spectrum(normalized_flux, normalized_ivar, vectorizer, theta, s2,
    **kwargs):
    """
    Solve the labels for given pixel fluxes and uncertainties for a single star.

    :param normalized_flux:
        The normalized fluxes. These should be on the same dispersion scale
        as the trained data.

    :param normalized_ivar:
        The 1-sigma uncertainties in the fluxes. This should have the same
        shape as `normalized_flux`.

    :param vectorizer:
        The model vectorizer.

    :param theta:
        The theta coefficients obtained from the training phase.

    :param s2:
        The intrinsic pixel variance.

    :returns:
        The labels and covariance matrix.
    """

    """
    # TODO: Re-visit this.
    # Get an initial estimate of the label vector from a matrix inversion,
    # and then ask the vectorizer to interpret that label vector into the 
    # (approximate) values of the labels that could have produced that 
    # label vector.
    lv = _estimate_label_vector(theta, s2, normalized_flux, normalized_ivar)
    initial = vectorizer.get_approximate_labels(lv)
    """

    # Overlook the bad pixels.
    inv_var = normalized_ivar/(1. + normalized_ivar * s2)
    use = np.isfinite(inv_var * normalized_flux)

    kwds = {
        "p0": vectorizer.fiducials,
        "maxfev": 10**6,
        "sigma": np.sqrt(1.0/inv_var[use]),
        "absolute_sigma": True
    }
    kwds.update(kwargs)
    
    f = lambda t, *l: np.dot(t, vectorizer(l).T).flatten()
    labels, cov = op.curve_fit(f, theta[use], normalized_flux[use], **kwds)
    return (labels, cov)


def _fit_pixel(normalized_flux, normalized_ivar, scatter, design_matrix,
    fixed_scatter=False, **kwargs):
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

    :param scatter:
        Fit the data using a fixed scatter term. If this value is set to None,
        then the scatter will be calculated.

    :returns:
        The optimised label vector coefficients and scatter for this pixel, even
        if it was supplied by the user.
    """

    if isinstance(design_matrix, string_types):
        with open(design_matrix, "rb") as fp:
            design_matrix = pickle.load(fp)

    # This initial theta will also be returned if we have no valid fluxes.
    initial_theta = np.hstack([1, np.zeros(design_matrix.shape[1] - 1)])

    if np.all(normalized_ivar == 0):
        return np.hstack([initial_theta, scatter if fixed_scatter else 0])

    # Optimize the parameters.
    kwds = {
        "maxiter": np.inf,
        "maxfun": np.inf,
        "disp": False,
        "full_output": True
    }
    kwds.update(kwargs.get("op_kwargs", {}))
    args = (normalized_flux, normalized_ivar, design_matrix)    
    logger.debug("Optimizer kwds: {}".format(kwds))

    if fixed_scatter:
        p0 = initial_theta
        func = _model_pixel_fixed_scatter
        args = tuple([scatter] + list(args))

    else:
        p0 = np.hstack([initial_theta, p0_scatter])
        func = _model_pixel

    op_params, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        func, p0, args=args, **kwds)

    if warnflag > 0:
        logger.warning("Warning: {}".format([
            "Maximum number of function evaluations made during optimisation.",
            "Maximum number of iterations made during optimisation."
            ][warnflag - 1]))

    return np.hstack([op_params, scatter]) if fixed_scatter else op_params


def _fit_pixel_s2_theta_separately(normalized_flux, normalized_ivar, scatter, design_matrix,
    fixed_scatter=False, **kwargs):

    """
    theta, ATCiAinv, inv_var = _fit_theta(normalized_flux, normalized_ivar,
        scatter, design_matrix)

    # Singular matrix or fixed scatter?
    if ATCiAinv is None or fixed_scatter:
        return np.hstack([theta, scatter if fixed_scatter else 0.0])

    # Optimise the pixel scatter, and at each pixel scatter value we will 
    # calculate the optimal vector coefficients for that pixel scatter value.
    kwds = {
        "maxiter": np.inf,
        "maxfun": np.inf,
        "disp": False, 
        "full_output":True

    }
    kwds.update(kwargs.get("op_kwargs", {}))
    logger.info("Passing to optimizer: {}".format(kwds))

    op_scatter, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        _fit_pixel_with_fixed_scatter, scatter,
        args=(normalized_flux, normalized_ivar, design_matrix),
        **kwds)

    if warnflag > 0:
        logger.warning("Warning: {}".format([
            "Maximum number of function evaluations made during optimisation.",
            "Maximum number of iterations made during optimisation."
            ][warnflag - 1]))

    theta, ATCiAinv, inv_var = _fit_theta(normalized_flux, normalized_ivar,
        op_scatter, design_matrix)
    return np.hstack([theta, op_scatter])
    """


def _model_pixel(theta, scatter, normalized_flux, normalized_ivar,
    design_matrix, **kwargs):

    inv_var = normalized_ivar/(1. + normalized_ivar * scatter**2)
    return model._chi_sq(theta, design_matrix, normalized_flux, inv_var) 
         #+ model._log_det(inv_var)


def _model_pixel_fixed_scatter(parameters, normalized_flux, normalized_ivar,
    design_matrix, **kwargs):
    
    theta, scatter = parameters[:-1], parameters[-1]
    return _model_pixel(
        theta, scatter, normalized_flux, normalized_ivar, design_matrix)


def _fit_pixel_with_fixed_scatter(scatter, normalized_flux, normalized_ivar,
    design_matrix, **kwargs):
    """
    Fit the normalized flux for a single pixel (across many stars) given some
    pixel variance term, and return the best-fit theta coefficients.

    :param scatter:
        The additional scatter to adopt in the pixel.

    :param normalized_flux:
        The normalized flux values for a single pixel across many stars.

    :param normalized_ivar:
        The inverse variance of the normalized flux values for a single pixel
        across many stars.

    :param design_matrix:
        The design matrix for the model.
    """

    theta, ATCiAinv, inv_var = _fit_theta(normalized_flux, normalized_ivar,
        scatter, design_matrix)

    return_theta = kwargs.get("__return_theta", False)
    if ATCiAinv is None:
        return 0.0 if not return_theta else (0.0, theta)

    # We take inv_var back from _fit_theta because it is the same quantity we 
    # need to calculate, and it saves us one operation.
    Q   = model._chi_sq(theta, design_matrix, normalized_flux, inv_var) 
        #+ model._log_det(inv_var)
    return (Q, theta) if return_theta else Q


def _fit_theta(normalized_flux, normalized_ivar, scatter, design_matrix):
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

    ivar = normalized_ivar/(1. + normalized_ivar * scatter**2)
    CiA = design_matrix * np.tile(ivar, (design_matrix.shape[1], 1)).T
    try:
        ATCiAinv = np.linalg.inv(np.dot(design_matrix.T, CiA))
    except np.linalg.linalg.LinAlgError:
        #if logger.getEffectiveLevel() == logging.DEBUG: raise
        return (np.hstack([1, [0] * (design_matrix.shape[1] - 1)]), None, ivar)

    ATY = np.dot(design_matrix.T, normalized_flux * ivar)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv, ivar)

