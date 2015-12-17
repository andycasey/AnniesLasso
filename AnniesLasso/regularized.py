#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A regularized (compressed sensing) version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["L1RegularizedCannonModel"]

import logging
import cPickle as pickle
import numpy as np
import multiprocessing as mp
import scipy.optimize as op
from sys import stdout, maxsize
from six import string_types

from . import (cannon, model, utils)

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


    def train(self, fixed_scatter=False, progressbar=True, initial_theta=None,
        **kwargs):
        """
        Train the model based on the labelled set using the given vectorizer.

        :param fixed_scatter: [optional]
            Fix the scatter terms and do not solve for them during the training
            phase. If set to `True`, the `s2` attribute must be already set.

        :param progressbar: [optional]
            Show a progress bar.
        """

        print("OVERWRITING FIXED SCATTER AND S2")
        self.s2 = 0.0
        fixed_scatter = True

        kwds = {
            "fixed_scatter": fixed_scatter,
            "progressbar": progressbar,
        }
        kwds.update(kwargs)

        if initial_theta is None or initial_theta is True:
            initial_theta = [initial_theta] * self.dispersion.size

        # This is a hack for speed: if regularization is zero, things are fast!
        if not np.all(self.regularization == 0):
            kwds.update({
                "function": _fit_regularized_pixel,
                "additional_args": [self.regularization, initial_theta]
            })

        logger.debug("SENDING: {}".format(kwds))
        super(L1RegularizedCannonModel, self).train(**kwds)


    def validate_regularization(self, fixed_scatter=False, Lambdas=None,
        pixel_mask=None, mod=10, **kwargs):
        """
        Perform validation upon several regularization parameters for each pixel
        using a subset of the labelled data set.

        :param fixed_scatter: [optional]
            Keep a fixed scatter term when doing the regularization validation.

        :param Lambdas: [optional]
            The regularization factors to evaluate. If `None` is specified, a
            sensible range will be automagically chosen.

        :param pixel_mask: [optional]
            An optional mask to only perform the regularization validation on.
            If given, a `False` entry indicates a pixel will not be evaluated.

        :param mod: [optional]
            The number of components to split the labelled set up into.

        :param kwargs: [optional]   
            These keyword arguments will be passed directly to the `train()`
            method.
        """

        if fixed_scatter and self.s2 is None:
            raise ValueError("intrinsic pixel variance (s2) must be set "
                             "before training if fixed_scatter is set to True")

        overwrite = kwargs.pop("overwrite", False)
        include_training_data = kwargs.pop("include_training_data", False)
        model_filename_format = kwargs.pop("model_filename_format", None)

        if Lambdas is None:
            Lambdas = np.hstack([0, 10**np.arange(0, 10.1, 0.1)])

        if pixel_mask is None:
            pixel_mask = np.ones_like(self.dispersion, dtype=bool)
            normalized_flux, normalized_ivar, dispersion = \
                (self.normalized_flux, self.normalized_ivar, self.dispersion)

        else:
            # Apply pixel masks now so we don't have to N_regularization times
            dispersion = self.dispersion[pixel_mask]
            normalized_flux = self.normalized_flux[:, pixel_mask]
            normalized_ivar = self.normalized_ivar[:, pixel_mask]
            
        # Determine the train and validate component masks.
        subsets = self._metadata["q"] % mod
        train_set, validate_set = (subsets > 0, subsets == 0)
        N_train, N_validate = map(sum, (train_set, validate_set))

        train_labelled_set = self.labelled_set[train_set]
        train_normalized_flux = normalized_flux[train_set]
        train_normalized_ivar = normalized_ivar[train_set]

        validate_normalized_flux = normalized_flux[validate_set]
        validate_normalized_ivar = normalized_ivar[validate_set]
        
        N_px, N_Lambdas = dispersion.size, len(Lambdas)

        models = []
        chi_sq = np.zeros((N_Lambdas, N_px))
        log_det = np.zeros((N_Lambdas, N_px))
        previous_theta = kwargs.pop("initial_theta", [None])
        if previous_theta is None:
            previous_theta = [None]
        for i, Lambda in enumerate(Lambdas):
            logger.info("Setting Lambda = {0}".format(Lambda))

            # Set up a model for this Lambda test.
            m = self.__class__(train_labelled_set, train_normalized_flux, 
                train_normalized_ivar, dispersion=dispersion, copy=False,
                threads=1 if self.pool is None else self.pool._processes)
            m.vectorizer = self.vectorizer
            m.regularization = Lambda
            if fixed_scatter:
                m.s2 = self.s2[pixel_mask]

            assert fixed_scatter and np.all(m.s2 == 0)

            # We want to make sure that we have the same training set each time.
            m._metadata.update({ "q": self._metadata["q"], "mod": mod })
            logger.info("SENDING PREVIOUS THETA: {}".format(previous_theta[-1]))
            m.train(
                fixed_scatter=fixed_scatter, 
                initial_theta=previous_theta[-1],
                **kwargs)
            #logger.info("Not sending previous theta")

            previous_theta.append(m.theta)
            if m.pool is not None: m.pool.close()

            if model_filename_format is not None:
                # Update the model to include train + validate in case we save
                # it with the data..
                m._normalized_flux = normalized_flux
                m._normalized_ivar = normalized_ivar
                m._labelled_set = self.labelled_set
                m.save(model_filename_format.format(i), overwrite=overwrite,
                    include_training_data=include_training_data)

            # Predict the fluxes in the validate set.
            inv_var = validate_normalized_ivar / \
                (1. + validate_normalized_ivar * m.s2)
            design_matrix = m.vectorizer(np.vstack(
                [self.labelled_set[label_name][validate_set] \
                    for label_name in self.vectorizer.label_names]).T)

            # Save everything.
            #chi_sq[i, :] = model._chi_sq(m.theta, design_matrix,
            #    validate_normalized_flux.T, inv_var.T, axis=1)
            for j in range(N_px):
                chi_sq[i, j] = np.sum(inv_var[:, j] * (np.dot(m.theta[j], design_matrix.T) - validate_normalized_flux[:, j].T)**2)
            #chi_sq[i, :] = inv_var.T * (np.dot(m.theta, design_matrix.T) - validate_normalized_flux.T)**2
            #log_det[i, :] = model._log_det(inv_var)
            models.append(m)
    
        return (Lambdas, chi_sq, log_det, models)


    def cross_validate_regularization(self, Lambdas, fixed_scatter=False,
        pixel_mask=None, mod=10, **kwargs):
        # Leave out 2/10ths of the data.

        # do the lambdas on 7/10ths of the data.
        
        # chose the best lambda per pixel using the predictions from 1/10th of
        # the data

        # fix those lambdas and train the model using 7/10ths of the data.
        # predict the labels for the remaining 1/10th of the data.

        # fix lambda = 0 and train a normal model using 7/10ths of the data.
        # predict the labels for the remaining 1/10th of the data.

        if fixed_scatter and self.s2 is None:
            raise ValueError("intrinsic pixel variance (s2) must be set "
                             "before training if fixed_scatter is set to True")

        if pixel_mask is None:
            pixel_mask = np.ones_like(self.dispersion, dtype=bool)
            normalized_flux, normalized_ivar, dispersion = \
                (self.normalized_flux, self.normalized_ivar, self.dispersion)

        else:
            # Apply pixel masks now so we don't have to N_regularization times
            dispersion = self.dispersion[pixel_mask]
            normalized_flux = self.normalized_flux[:, pixel_mask]
            normalized_ivar = self.normalized_ivar[:, pixel_mask]

        # Should we save progress?
        model_filename_format = kwargs.pop("model_filename_format", None)
        overwrite = kwargs.pop("overwrite", False)
        include_training_data = kwargs.pop("include_training_data", False)


        # Determine the train and validate component masks.
        subsets = self._metadata["q"] % mod
        validate_set = (subset == 0)
        train_set = (subset > 0) * (subset < (mod - 1))
        test_set = (subset == (mod - 1))

        N_train, N_validate, N_test = map(sum, (train_set, validate_set, test_set))

        train_labelled_set = self.labelled_set[train_set]
        train_normalized_flux = normalized_flux[train_set]
        train_normalized_ivar = normalized_ivar[train_set]

        validate_normalized_flux = normalized_flux[validate_set]
        validate_normalized_ivar = normalized_ivar[validate_set]
        
        N_Lambdas = len(Lambdas), len(dispersion)

        models = []
        Q = np.zeros((N_Lambdas, N_px))
        
        for i, Lambda in enumerate(Lambdas):
            logger.info("Setting Lambda = {0}".format(Lambda))

            # Set up a model for this Lambda test.
            m = self.__class__(train_labelled_set, train_normalized_flux, 
                train_normalized_ivar, dispersion=dispersion, copy=False,
                threads=1 if self.pool is None else self.pool._processes)
            m.vectorizer = self.vectorizer
            m.regularization = Lambda
            if fixed_scatter:
                m.s2 = self.s2[pixel_mask]

            # We want to make sure that we have the same training set each time.
            m._metadata.update({ "q": self._metadata["q"], "mod": mod })

            m.train(fixed_scatter=fixed_scatter)
            if m.pool is not None: m.pool.close()

            if model_filename_format is not None:
                # Update the model to include train + validate in case we save
                # it with the data..
                m._dispersion = dispersion
                m._normalized_flux = normalized_flux
                m._normalized_ivar = normalized_ivar
                m._labelled_set = self.labelled_set
                m.save(model_filename_format.format(i),
                    overwrite=overwrite, include_training_data=include_training_data)

            # Predict the fluxes in the validate set.
            inv_var = validate_normalized_ivar / \
                (1. + validate_normalized_ivar * m.s2)
            design_matrix = m.vectorizer(np.vstack(
                [self.labelled_set[label_name][validate_set] \
                    for label_name in self.vectorizer.label_names]).T)

            # Save everything.
            Q[i, :] = model._chi_sq(m.theta, design_matrix,
                validate_normalized_flux.T, inv_var.T, axis=1)
                #+ model._log_det(inv_var)
            models.append(m)

        Q = Q - Q[0]
        Q /= validate_set.sum()

        # Get the best Lambda value at each Q.
        indices = np.argmin(Q, axis=1)
        Lambda_opt = Lambda[indices]
        logger.debug("Lambda_opt: {}".format(Lambda_opt))

        # Train a model with the train data using the best Lambda value for each
        # pixel.
        regularized_model = self.__class__(
            train_labelled_set, train_normalized_flux, train_normalized_ivar,
            dispersion=dispersion, copy=False,
            threads=1 if self.pool is None else self.pool._processes)
        regularized_model.vectorizer = self.vectorizer
        regularized_model.regularization = Lambda_opt
        regularized_model.train()

        # Train a model with the train data using no regularization.
        unregularized_model = self.__class__(
            train_labelled_set, train_normalized_flux, train_normalized_ivar,
            dispersion=dispersion, copy=False,
            threads=1 if self.pool is None else self.pool._processes)
        unregularized_model.vectorizer = self.vectorizer
        unregularized_model.regularization = 0.0
        unregularized_model.train()

        # Predict the labels for stars in the test step using both models.
        regularized_predicted_labels = regularized_model.fit(
            self.normalized_flux[test_set],
            self.normalized_ivar[test_set])

        unregularized_predicted_labels = regularized_model.fit(
            self.normalized_flux[test_set],
            self.normalized_ivar[test_set])

        # How do they compare to the actual labels?
        expected_labels = self.labels_array[test_set]

        # SAVE EVERYTHING
        dumpfile = kwargs.pop("dumpfile", "CV_REGULARIZATION_DUMP_FILE.PKL")
        with open(dumpfile, "wb") as fp:
            pickle.dump((self, Lambda, Q, regularized_model, unregularized_model,
                regularized_predicted_labels, unregularized_predicted_labels,
                expected_labels))

        logger.info("Unregularized model:")
        difference = unregularized_predicted_labels - expected_labels
        for i, label_name in enumerate(self.vectorizer.label_names):
            l, m, u = np.percentile(difference[:, i])
            logger.info("DELTA({0}): {1:.3f} ({2:.3f}, {3:.3f})".format(
                label_name, m, m - l, u - m))

        logger.info("Regularized model:")
        difference = regularized_predicted_labels - expected_labels
        for i, label_name in enumerate(self.vectorizer.label_names):
            l, m, u = np.percentile(difference[:, i])
            logger.info("DELTA({0}): {1:.3f} ({2:.3f}, {3:.3f})".format(
                label_name, m, m - l, u - m))
        

        raise a



def chi_sq(theta, design_matrix, data, inv_var, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and data.
    """
    residuals = np.dot(theta, design_matrix.T) - data

    f = np.sum(inv_var * residuals**2, axis=axis)
    if not gradient:
        return f

    g = 2.0 * np.dot(inv_var * residuals, design_matrix)
    return (f, g)

    
def L1Norm(Q):
    """
    Return the L1 normalization of Q.

    :param Q:
        An array of finite values.
    """
    return (np.sum(np.abs(Q)), np.sign(Q))


def _fit_pixel_with_fixed_regularization_and_fixed_scatter(theta, scatter,
    normalized_flux, normalized_ivar, regularization, design_matrix):
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

    inv_var = normalized_ivar/(1. + normalized_ivar * scatter**2)
    csq, d_csq = chi_sq(theta, design_matrix, normalized_flux, inv_var)
    L1, d_L1 = L1Norm(theta)

    # We are using a variation of L1 norm that ignores the first coefficient.
    L1 = L1 - np.abs(theta[0])

    f = csq + regularization * L1
    g = d_csq + regularization * d_L1
    return (f, g)



def _fit_pixel_with_fixed_regularization(parameters, normalized_flux,
    normalized_ivar, regularization, design_matrix):
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
    theta, scatter = parameters[:-1], parameters[-1]
    return _fit_pixel_with_fixed_regularization_and_fixed_scatter(
        theta, scatter, normalized_flux, normalized_ivar, regularization,
        design_matrix)




def _fit_regularized_pixel(normalized_flux, normalized_ivar, scatter,
    regularization, initial_theta, design_matrix, fixed_scatter=False, 
    **kwargs):

    if isinstance(design_matrix, string_types):
        with open(design_matrix, "rb") as fp:
            design_matrix = pickle.load(fp)


    # Any actual information?
    if np.all(normalized_ivar == 0):
        return_theta = np.hstack([1, np.zeros(design_matrix.shape[1] - 1)])
        return np.hstack([return_theta, scatter if fixed_scatter else 0])

    if initial_theta is None:
        initial_theta = np.hstack([1, np.zeros(design_matrix.shape[1] - 1)])

    assert initial_theta is not True
    #elif initial_theta is True:
    #    initial_theta, _, __ = cannon._fit_theta(
    #        normalized_flux, normalized_ivar, scatter, design_matrix)

    logger.debug("Using initial theta: {}".format(initial_theta))
    if fixed_scatter:
        p0 = initial_theta
        func = _fit_pixel_with_fixed_regularization_and_fixed_scatter
        args = (scatter, normalized_flux, normalized_ivar, regularization,
            design_matrix)
    else:
        p0 = np.hstack([initial_theta, scatter])
        func = _fit_pixel_with_fixed_regularization
        args = (normalized_flux, normalized_ivar, regularization, design_matrix)

    # Prepare keywords for optimization.
    kwds = {
        "args": args,
        "disp": False,
        "maxfun": np.inf,
        "maxiter": np.inf,
    }
    bfgs_terms = kwargs.pop("op_bfgs_kwargs", {})
    kwds.update(bfgs_terms)

    logger.debug("Checking that kwds are going in to bFGS: {}".format(kwds))
    assert scatter == 0.0

    f = lambda theta: my_func(theta, design_matrix, normalized_flux, normalized_ivar, regularization)
    g = lambda theta: my_grad(theta, design_matrix, normalized_flux, normalized_ivar, regularization)

    op_params, fopt, d = op.fmin_l_bfgs_b(func, p0, fprime=None,
        approx_grad=False, **kwds)

    if d["warnflag"] > 0:
        logger.warning("BFGS stopped prematurely: {}".format(d["task"]))

        # Run Powell's method instead.
        #xtol, ftol = kwargs.get(("xtol", "ftol"), (1e-4, 1e-4))
        kwds.update(kwargs.get("op_kwargs", {}))
        for k in bfgs_terms:
            del kwds[k]

        #print("using ftol xtol {0} {1}".format(xtol, ftol))
        """
        op_params, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
            func, op_params, full_output=True, **kwds)

        if warnflag > 0:
            logger.warn("Powell optimization failed: {}".format([
                    "Maximum number of function evaluations.",
                    "Maximum number of iterations."
                ][warnflag - 1]))
        else:
            logger.info("Powell optimization completed successfully.")
        """

    return np.hstack([op_params, scatter]) if fixed_scatter else op_params



'''

def _fit_regularized_pixel_incorrectly(normalized_flux, normalized_ivar, scatter,
    regularization, design_matrix, fixed_scatter=False, **kwargs):
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

    #design_matrix, kwargs = cannon._get_design_matrix(normalized_flux.shape[0], **kwargs)

    theta, ATCiAinv, inv_var = cannon._fit_theta(
        normalized_flux, normalized_ivar, scatter, design_matrix)

    # Singular matrix?
    if ATCiAinv is None:
        return np.hstack([theta, scatter if fixed_scatter else np.inf])

    if fixed_scatter:
        p0 = theta
        func = _fit_pixel_with_fixed_regularization_and_fixed_scatter
        args = (normalized_flux, inv_var, scatter, regularization,
            design_matrix)
    else:
        p0 = np.hstack([scatter, theta])
        func = _fit_pixel_with_fixed_regularization
        args = (normalized_flux, normalized_ivar, regularization, design_matrix)

    kwds = { "disp": False, "maxiter": np.inf, "maxfun": np.inf }
    #kwds.update(kwargs)
    kwds.update(kwargs.get("op_kwargs", {}))

    logger.info("Passing to optimizer: {}".format(kwds))

    logger.debug("Optimizing pixel from {0} {1} (fixed_scatter = {2})".format(
        scatter, theta, fixed_scatter))
    parameters, fopt, direc, n_iter, n_funcalls, warnflag = op.fmin_powell(
        func, p0, args=args, full_output=True, retall=False, **kwds)
    if not fixed_scatter: scatter, parameters = parameters[0], parameters[1:]
    
    if warnflag > 0:
        stdout.write("\r\n")
        stdout.flush()
        logger.warning("Optimization stopped prematurely: {}".format([
            "Maximum number of function evaluations.",
            "Maximum number of iterations."
            ][warnflag - 1]))

    logger.debug("Optimized result: {0} {1} (fixed_scatter = {2})".format(
        parameters, scatter, fixed_scatter))

    return np.hstack([parameters, scatter])
'''

