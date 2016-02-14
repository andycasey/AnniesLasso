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
import tempfile
import os

from . import (model, utils)

logger = logging.getLogger(__name__)


class CannonModel(model.BaseCannonModel):
    """
    A generalised Cannon model for the estimation of arbitrary stellar labels.

    :param labelled_set:
        A set of labelled objects. The most common input form is a table with
        columns as labels, and stars/objects as rows.

    :type labelled_set:
        :class:`~astropy.table.Table` or a numpy structured array

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

    :type dispersion:
        :class:`np.array`

    :param threads: [optional]
        Specify the number of parallel threads to use. If `threads > 1`, the
        training and prediction phases will be automagically parallelised.

    :type threads:
        int

    :param pool: [optional]
        Specify an optional multiprocessing pool to map jobs onto.
        This argument is only used if specified and if `threads > 1`.
    
    :type pool:
        bool
    """
    def __init__(self, *args, **kwargs):
        super(CannonModel, self).__init__(*args, **kwargs)


    @model.requires_model_description
    def train(self, fixed_scatter=True, **kwargs):
        """
        Train the model based on the labelled set.
        """
        
        # Experimental/asthetic keywords:
        # use_neighbouring_pixels, progressbar
        assert fixed_scatter, "Are you refactoring?"
        if self.s2 is None:
            logger.warn("Fixing and assuming s2 = 0")
            self.s2 = 0

        if fixed_scatter and self.s2 is None:
            raise ValueError("intrinsic pixel variance (s2) must be set "
                             "before training if fixed_scatter is set to True")

        # We default use_neighbouring_pixels to None so that we can default it
        # to True later if we want, but we can warn the user if they explicitly
        # set it to True and we intend to ignore it.
        use_neighbouring_pixels = kwargs.pop("use_neighbouring_pixels", None)
        if self.theta is None:
            if use_neighbouring_pixels is None:
                use_neighbouring_pixels = True
            initial_theta = [None] * self.dispersion.size

        else:
            # Since theta is already set, we will ignore neighbouring pixels.
            if use_neighbouring_pixels is True:
                use_neighbouring_pixels = False
                logger.warn("Ignoring neighbouring pixels because theta is "
                            "already provided.")
            initial_theta = self.theta.copy()
    
        # Initialize the scatter.
        initial_s2 = self.s2 if fixed_scatter \
            else 0.01**2 * np.ones_like(self.dispersion)

        # Prepare the method and arguments.
        fitting_function = kwargs.pop("function", _fit_pixel)
        kwds = {
            "fixed_scatter": fixed_scatter,
            "design_matrix": self.design_matrix,
            "op_kwargs": kwargs.pop("op_kwargs", {}),
            "op_bfgs_kwargs": kwargs.pop("op_bfgs_kwargs", {})
        }

        N_stars, N_pixels = self.normalized_flux.shape
        logger.info("Training {0}-label {1} with {2} stars and {3} pixels/star"\
            .format(len(self.vectorizer.label_names), type(self).__name__,
                N_stars, N_pixels))

        # Arguments:
        # initial_theta, initial_s2, flux, ivar, [additional_args], 
        # design_matrix, **kwargs
        args = [initial_theta, initial_s2, self.normalized_flux.T, 
            self.normalized_ivar.T]
        args.extend(kwargs.get("additional_args", []))

        # Write the design matrix to a temporary file.
        temporary_filenames = []
        #temporary_filename = utils._pack_value(self.design_matrix)
        #kwds["design_matrix"] = temporary_filename
        #temporary_filenames.append(temporary_filename)

        # Wrap the function so we can parallelize it out.
        mapper = map if self.pool is None else self.pool.map
        try:
            f = utils.wrapper(fitting_function, None, kwds, N_pixels)
            if self.pool is None and use_neighbouring_pixels:
                output = []
                neighbour_theta = []
                for j, row in enumerate(zip(*args)):
                    if j > 0:
                        # Update with determined theta from neighbour pixel.
                        row = list(row)
                        row[0] = neighbour_theta
                        row = tuple(row)
                    output.append(f(*row))
                    neighbour_theta = output[-1][0][:design_matrix.shape[1]]

            else:
                output = mapper(f, [row for row in zip(*args)])

        except KeyboardInterrupt:
            logger.debug("Removing temporary filenames:\n{}".format(
                "\n".join(temporary_filenames)))
            map(os.remove, temporary_filenames)

        else:
            results = []
            metadata = []
            for r, m in output:
                results.append(r)
                metadata.append(m)

            results = np.array(results)

        # Clean up any temporary files.
        for filename in temporary_filenames:
            if os.path.exists(filename): os.remove(filename)

        # Unpack the results.
        self.theta, self.s2 = (results[:, :-1], results[:, -1])

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
    def fit(self, normalized_flux, normalized_ivar, full_output=False, **kwargs):
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
        N_spectra = normalized_flux.shape[0]
        
        # Prepare the wrapper function and data.
        message = None if not kwargs.pop("progressbar", True) \
            else "Fitting {0} spectra".format(N_spectra)

        f = utils.wrapper(
            _fit_spectrum, (self.vectorizer, self.theta, self.s2), kwargs, 
            N_spectra, message=message)

        args = (normalized_flux, normalized_ivar)
        mapper = map if self.pool is None else self.pool.map

        labels, cov, metadata = zip(*mapper(f, zip(*args)))
        labels, cov = map(np.array, (labels, cov))

        return (labels, cov, metadata) if full_output else labels


    @model.requires_training_wheels
    def _set_s2_by_hogg_heuristic(self):
        """
        Set the pixel scatter by Hogg's heuristic.

        See https://github.com/andycasey/AnniesLasso/issues/31 for more details.
        """

        model_flux = self.predict(self.labels_array)
        residuals_squared = (model_flux - self.normalized_flux)**2

        def objective_function(s, residuals_squared, ivar):
            adjusted_ivar = ivar/(1. + ivar * s**2)
            chi_sq = residuals_squared * adjusted_ivar
            return (np.mean(chi_sq) - 1.0)**2

        s = []
        for j in range(self.dispersion.size):
            print(j)
            s.append(op.fmin(objective_function, 0,
                args=(residuals_squared[:, j], self.normalized_ivar[:, j]), 
                disp=False))

        s2 = np.array(s)**2
        s2[s2 == 0] = np.inf
        self.s2 = s2
        return True


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
    Fit a single spectrum by least-squared fitting.
    
    :param normalized_flux:
        The normalized flux values.

    :param normalized_ivar:
        The inverse variance array for the normalized fluxes.

    :param vectorizer:
        The vectorizer to use when fitting the data.

    :param theta:
        The theta coefficients (spectral derivatives) of the trained model.

    :param s2:
        The pixel scatter (s^2) array for each pixel.
    """

    adjusted_ivar = normalized_ivar/(1. + normalized_ivar * s2)
    adjusted_inv_sigma = np.sqrt(adjusted_ivar)

    def objective(labels):
        model_flux = np.dot(theta, vectorizer(labels).T).flatten()
        return adjusted_inv_sigma * (model_flux - normalized_flux)

    # Check the vectorizer whether it has a derivative built in.
    try:
        vectorizer.get_label_vector_derivative(vectorizer.fiducials)
    
    except NotImplementedError:
        logger.debug("No label vector derivative available!")
        Dfun = None

    except:
        logger.exception("Exception raised when trying to calculate the label "
                         "vector derivative at the fiducial values:")
        raise

    else:
        # Use the label vector derivative.
        # Presumably because of the way leastsq works, the adjusted_inv_sigma
        # does not enter here, otherwise we get incorrect results.
        
        # TODO: Revisit this with Hogg.
        #Dfun = lambda labels: \
        #    np.dot(theta, vectorizer.get_label_vector_derivative(labels)).T
        Dfun = None

    kwds = {
        "func": objective,
        "x0": vectorizer.fiducials,
        "args": (),
        "Dfun": Dfun,
        "col_deriv": True,
        "ftol": 7./3 - 4./3 - 1, # Machine precision.
        "xtol": 7./3 - 4./3 - 1, # Machine precision.
        "gtol": 0.0,
        "maxfev": int(2e4), # MAGIC
        "epsfcn": None,
        "factor": 0.1, # Smallest step size available for gradient approximation
        "diag": 1.0/vectorizer.scales
    }
    # Only update the keywords with things that op.leastsq expects.
    kwds.update({ k: v for k, v in kwargs.items() if k in kwds })
    op_labels, cov, meta, message, ier = op.leastsq(full_output=True, **kwds)

    if ier not in range(1, 5):
        logger.warn("Least-sq result was {0}: {1}".format(ier, message))

    # Save additional information.
    _ = np.sum(meta["fvec"]**2)
    meta.update({ 
        "method": "leastsq",
        "ier": ier,
        "message": message,
        "Dfun": Dfun is not None,
        "chi_sq": _,
        "r_chi_sq": _/(normalized_flux.size - len(vectorizer.fiducials) - 1),
        "model_flux": np.dot(theta, vectorizer(op_labels).T).flatten()
    })
    meta.update({ k: kwds[k] for k in \
        ("ftol", "xtol", "gtol", "maxfev", "factor", "epsfcn") })

    return (op_labels, cov, meta)


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

    design_matrix = utils._unpack_value(design_matrix)

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
        scatter**2, design_matrix)

    return_theta = kwargs.get("__return_theta", False)
    if ATCiAinv is None:
        return 0.0 if not return_theta else (0.0, theta)

    # We take inv_var back from _fit_theta because it is the same quantity we 
    # need to calculate, and it saves us one operation.
    Q   = model._chi_sq(theta, design_matrix, normalized_flux, inv_var) 
    return (Q, theta) if return_theta else Q


def _fit_theta(normalized_flux, normalized_ivar, s2, design_matrix):
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

    ivar = normalized_ivar/(1. + normalized_ivar * s2)
    CiA = design_matrix * np.tile(ivar, (design_matrix.shape[1], 1)).T
    try:
        ATCiAinv = np.linalg.inv(np.dot(design_matrix.T, CiA))
    except np.linalg.linalg.LinAlgError:
        #if logger.getEffectiveLevel() == logging.DEBUG: raise
        return (np.hstack([1, [0] * (design_matrix.shape[1] - 1)]), None, ivar)

    ATY = np.dot(design_matrix.T, normalized_flux * ivar)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv, ivar)

