#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A regularized (compressed sensing) version of The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import logging
import numpy as np
import scipy.optimize as op

import tensorflow as tf

from .base import BaseCannonModel

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
        return None



    def train(self, threads=None):


        for i in range(self.training_set_flux.shape[1]):

            print(" ON {}".format(i))

            initial_theta = np.hstack([1.0, np.zeros(len(self.vectorizer.terms))])

            initial_theta, ATCiAinv, adjusted_ivar = _fit_theta_by_linalg(
                self.training_set_flux[:, i], 
                self.training_set_ivar[:, i], 
                s2=0, design_matrix=self.design_matrix)

            # TODO: Check if the fiducial theta is a better starting point?

            # Use a previously-trained value as the starting point?
            # Use a neighbouring value?
            if not np.all(np.isfinite(initial_theta)):
                continue

            foo = _fit_pixel_fixed_scatter(self.training_set_flux[:, i], self.training_set_ivar[:, i], 
                initial_theta, self.design_matrix, regularization=0.0, censoring_mask=None)

        raise NotImplementedError


    def test(self, threads=None):
        raise NotImplementedError


    def fit(self, *args, **kwargs):
        return self.test(*args, **kwargs)



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
        return (np.hstack([1, np.zeros(design_matrix.shape[1] - 1)]), None, adjusted_ivar)

    ATY = np.dot(design_matrix.T, flux * adjusted_ivar)
    theta = np.dot(ATCiAinv, ATY)

    return (theta, ATCiAinv, adjusted_ivar)




# TODO: This logic should probably go somewhere else.


def chi_sq(theta, design_matrix, flux, ivar, axis=None, gradient=True):
    """
    Calculate the chi-squared difference between the spectral model and flux.
    """
    residuals = np.dot(theta, design_matrix.T) - flux

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
        L1, d_L1 = L1Norm(theta)

        # We are using a variation of L1 norm that ignores the first coefficient.
        f = csq + regularization * (L1 - np.abs(theta[0]))
        g = d_csq + regularization * d_L1
        
        print("opt", f)
        return (f, g)

    else:
        csq = chi_sq(theta, design_matrix, flux, ivar, gradient=False)
        L1, d_L1 = L1Norm(theta)

        # We are using a variation of L1 norm that ignores the first coefficient.
        return csq + regularization * (L1 - np.abs(theta[0]))
        

def _fit_pixel_fixed_scatter(flux, ivar, initial_theta, design_matrix, 
    regularization, censoring_mask, **kwargs): 

    if np.sum(ivar) < 1.0 * ivar.size: # MAGIC
        metadata = dict(message="No pixel information.")
        return (initial_theta, metadata)


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


    #initial_theta = np.hstack([1, np.zeros(initial_theta.size - 1)])
    

    from time import time


    # Optimization keyword arguments.
    kwds = dict(args=(design_matrix, flux, ivar, regularization), disp=False,
        maxfun=np.inf, maxiter=np.inf, m=initial_theta.size, factr=10.0,
        pgtol=1e-6)
    kwds.update(kwargs.pop("op_bfgs_kwds", {}))

    ta = time()

    op_params, fopt, metadata = op.fmin_l_bfgs_b(
        _pixel_objective_function_fixed_scatter, initial_theta, 
        fprime=None, approx_grad=None, **kwds)

    t_opt = time() - ta

    metadata.update(dict(fopt=fopt, p0=initial_theta))


    import pystan as stan

    stan_model = stan.StanModel(model_code="""
        data {
            int<lower=1> S; // number of stars
            int<lower=1> T; // number of terms
            real flux[S];
            real flux_sigma[S];
            matrix[T, S] DM;
        }

        parameters {
            row_vector[T] theta;
        }

        model {
            flux ~ normal(theta * DM, flux_sigma);    
        }
        """)

    data = dict(S=flux.size, T=initial_theta.size,
        flux=flux, flux_sigma=ivar**-0.5, DM=design_matrix.T)

    ta = time()
    stan_params = stan_model.optimizing(data=data, init={"theta": initial_theta},
        iter=10000)
    t_stan = time() - ta

    raise a


    # TENSORFLOW TEST

    print("A")
    theta = tf.Variable(initial_theta.reshape((-1, 1)), trainable=True, name="theta")

    print("B")
    loss = tf.reduce_sum(tf.multiply(ivar, tf.square(tf.reduce_sum(tf.multiply(theta, design_matrix.T), 0) - flux)))


    #    loss = tf.reduce_sum(tf.mul(theta, design_matrix), 1) - flux)**2

    #loss = tf.einsum('i,i->', theta, design_matrix.T) - flux

    
    print("C")
    opt = tf.train.AdamOptimizer(0.1).minimize(loss)

    tb = time() 
    
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for i in range(1000):

            session.run(opt)
            lossed = session.run(loss)
            print("tf", i, lossed, fopt)

        done = session.run(theta)
        t_tf = time() - tb

        print(done, lossed)

        #print("TF", session.run(loss))
        #for i in range(100):
        #    print("TF", session.run([theta, loss]))


    raise a










    if metadata["warnflag"] > 0:
        logger.warn(
            "BFGS optimization stopped with message: {}".format(metadata["task"]))

    # De-censor the optimized parameters.
    if censoring_mask is not None:
        theta = np.zeros(censoring_mask.size)
        theta[censoring_mask] = op_params

    else:
        theta = op_params

    return (theta, metadata)
