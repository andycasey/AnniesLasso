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
    A model for The Cannon which includes L1 regularization and pixel censoring.
    """

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

        if self.training_set_flux is None or self.training_set_ivar is None:
            raise TypeError(
                "cannot train: training set spectra not saved with the model")

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
