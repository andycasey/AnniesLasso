#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["CannonModel"]

import logging
import multiprocessing as mp
import numpy as np
import scipy.optimize as op
from functools import wraps
from time import time

from . import (base, fitting, utils)

logger = logging.getLogger(__name__)


def requires_training(method):
    """
    A decorator for model methods that require training before being run.

    :param method:
        A method belonging to a sub-class of BaseCannonModel.
    """
    @wraps(method)
    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model requires training first")
        return method(model, *args, **kwargs)
    return wrapper


class CannonModel(base.BaseCannonModel):
    """
    A model for The Cannon which includes L1 regularization and censoring.
    """

    def __init__(self, training_set_labels, training_set_flux, training_set_ivar,
        vectorizer, dispersion=None, regularization=None, censors=None, **kwargs):
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
            A vectorizer to take input labels and produce a design matrix. This
            should be a sub-class of `vectorizer.BaseVectorizer`.

        :param dispersion: [optional]
            The dispersion values corresponding to the given pixels. If provided, 
            this should have a size of `num_pixels`.
        
        :param regularization: [optional]
            The strength of the L1 regularization. This should either be `None`,
            a float-type value for single regularization strength for all pixels,
            or a float-like array of length `num_pixels`.

        :param censors: [optional]
            A dictionary containing label names as keys and boolean censoring
            masks as values.
        """

        super(CannonModel, self).__init__(**kwargs)

        # Save the vectorizer.
        self._vectorizer = vectorizer

        self._dispersion = dispersion

        if training_set_labels is None and training_set_flux is None \
        and training_set_ivar is None:

            # Must be reading in a model that does not have training set data
            # saved. Therefore we need the scales and fiducials.
            try:
                self._scales = kwargs["scales"]
                self._fiducials = kwargs["fiducials"]

            except KeyError:
                raise TypeError("the model needs a training set")

            self._training_set_labels = None
            self._training_set_flux = None
            self._training_set_ivar = None
            self._design_matrix = None

        else:

            self._training_set_labels = np.array(
                [training_set_labels[ln] for ln in vectorizer.label_names]).T
            self._training_set_flux = np.atleast_2d(training_set_flux)
            self._training_set_ivar = np.atleast_2d(training_set_ivar)
        
            # Check that the flux and ivar are valid, and dispersion if given.
            self._verify_training_data()

            # Offset and scale the training set labels.
            self._scales = np.ptp(
                np.percentile(self.training_set_labels, [2.5, 97.5], axis=0), axis=0)
            self._fiducials = np.percentile(self.training_set_labels, 50, axis=0)

            # Create a design matrix.
            self._design_matrix = vectorizer(
                (self.training_set_labels - self._fiducials)/self._scales).T

        # Check the regularization and censoring.
        self.regularization = regularization
        self.censors = censors
        
        self.reset()

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

        if self.training_set_flux is None:
            raise TypeError("cannot train: no training set saved with the model")

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

        func = utils.wrapper(fitting.fit_pixel_fixed_scatter, None, kwargs, P)

        meta = []
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
            (pixel_theta, pixel_s2, pixel_meta), = mapper(func, [args])

            meta.append(pixel_meta)
            theta[pixel], s2[pixel] = (pixel_theta, pixel_s2)

        self._theta, self._s2 = (theta, s2)

        if pool is not None:
            pool.close()
            pool.join()

        return (theta, s2, meta)


    @requires_training
    def __call__(self, labels):
        """
        Return spectral fluxes, given the labels.

        :param labels:
            An array of stellar labels.
        """

        # Scale and offset the labels.
        scaled_labels = (np.atleast_2d(labels) - self._fiducials)/self._scales
        flux = np.dot(self.theta, self.vectorizer(scaled_labels)).T
        return flux[0] if flux.shape[0] == 1 else flux


    @requires_training
    def test(self, flux, ivar, initial_labels=None, threads=None, **kwargs):
        """
        Run the test step on spectra.

        :param flux:
            The (pseudo-continuum-normalized) spectral flux.

        :param ivar:
            The inverse variance values for the spectral fluxes.

        :param initial_labels: [optional]
            The initial labels to try for each spectrum. This can be a single
            set of initial values, or one set of initial values for each star.

        :param threads: [optional]
            The number of parallel threads to use.
        """

        if threads in (1, None):
            mapper, pool = (map, None)

        else:
            pool = mp.Pool(threads)
            mapper = pool.map

        flux, ivar = (np.atleast_2d(flux), np.atleast_2d(ivar))
        S, P = flux.shape

        if ivar.shape != flux.shape:
            raise ValueError("flux and ivar arrays must be the same shape")

        if initial_labels is None:
            initial_labels = self._fiducials

        initial_labels = np.atleast_2d(initial_labels)
        if initial_labels.shape[0] != S and len(initial_labels.shape) == 2:
            initial_labels = np.tile(initial_labels.flatten(), S)\
                             .reshape(S, -1, len(self._fiducials))

        func = utils.wrapper(fitting.fit_spectrum, 
            (self.vectorizer, self.theta, self.s2, self._fiducials, self._scales),
            kwargs, S, message="Fitting {} spectra".format(S))

        labels, cov, meta = zip(*mapper(func, zip(*(flux, ivar, initial_labels))))

        if pool is not None:
            pool.close()
            pool.join()

        return (np.array(labels), np.array(cov), meta)


    def _initial_theta(self, pixel_index, **kwargs):
        """
        Return a list of guesses of the spectral coefficients for the given
        pixel index. Initial values are sourced in the following preference
        order: 

            (1) a previously trained `theta` value for this pixel,
            (2) an estimate of `theta` using linear algebra,
            (3) a neighbouring pixel's `theta` value,
            (4) the fiducial value of [1, 0, ..., 0].

        :param pixel_index:
            The zero-indexed integer of the pixel.

        :returns:
            A list of initial theta guesses, and the source of each guess.
        """

        guesses = []

        if self.theta is not None:
            # Previously trained theta value.
            if np.all(np.isfinite(self.theta[pixel_index])):
                guesses.append((self.theta[pixel_index], "previously_trained"))

        # Estimate from linear algebra.
        theta, cov = fitting.fit_theta_by_linalg(
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
