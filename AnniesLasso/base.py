#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An abstract model class for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseCannonModel"]


import logging
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os import path
from six.moves import cPickle as pickle
from six import string_types
from scipy.spatial import Delaunay

from . import (censoring, )

logger = logging.getLogger(__name__)



class BaseCannonModel(object):
    """
    An abstract Cannon model object that implements convenience functions,
    data validation, and appropriate properties.
    """

    _trained_attributes = ("theta", "s2")
    _descriptive_attributes = \
        ("vectorizer", "censors", "regularization", "dispersion")
    _data_attributes = 
        ("training_set_labels", "training_set_flux", "training_set_ivar")

    # Initiation:

    def __init__(self, *args, **kwargs):
        return None

    """
    _trained_attributes = ["_s2", "_theta"]
    _descriptive_attributes = ["_dispersion", "_vectorizer", "_censors"]
    _data_attributes = ["_labelled_set", "_normalized_flux", "_normalized_ivar"]
    
    def __init__(self, labelled_set, normalized_flux, normalized_ivar,
        dispersion=None, threads=1, pool=None, copy=False, verify=True):

        if threads == 1:
            self.pool = None
        else:
            # Allow a negative to set the max number of threads.
            threads = None if threads < 0 else threads
            self.pool = pool or utils.InterruptiblePool(threads,
                initializer=utils._init_pool, initargs=(utils._counter, ))

        self._metadata = { "threads": threads }

        # Initialise descriptive attributes for the model and verify the data.
        for attribute in self._descriptive_attributes:
            setattr(self, attribute, None)

        # Load in the labelled set.
        if  labelled_set is None \
        and normalized_flux is None \
        and normalized_ivar is None:
            self.reset()
            return None

        # Initialize the data arrays.
        self._init_data_attributes(
            labelled_set, normalized_flux, normalized_ivar)

        self._dispersion = np.array(dispersion).flatten() \
            if dispersion is not None \
            else np.arange(self._normalized_flux.shape[1], dtype=int)

        # Initialise wavelength censoring.
        self._censors = censoring.CensorsDict(self)

        # Initialize a random, yet reproducible group of subsets.
        self._metadata["q"] = np.random.randint(0, 10, len(labelled_set))
        
        if copy:
            self._labelled_set, self._dispersion \
                = map(deepcopy, (self._labelled_set, self._dispersion))
            self._normalized_flux, self._normalized_ivar, \
                = map(deepcopy, (self._normalized_flux, self._normalized_ivar))

        if verify: self._verify_training_data()
        self.reset()

        return None
    """

    # Representations.

    def __str__(self):
        return "<{module}.{name} of {K} labels {trained}with a training set "\
               "of {N} stars each with {M} pixels>".format(
                    module=self.__module__,
                    name=type(self).__name__,
                    trained="trained " if self.is_trained else "",
                    K=self.training_set_labels.shape[1],
                    N=self.training_set_labels.shape[0], 
                    M=self.training_set_flux.shape[1])


    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(self.__module__, 
            type(self).__name__, hex(id(self)))

    # Model attributes that cannot be changed.

    @property
    def training_set_labels(self):
        """ Return the labels in the training set. """
        return self._training_set_labels


    @property
    def training_set_flux(self):
        """ Return the training set fluxes. """
        return self._training_set_flux


    @property
    def training_set_ivar(self):
        """ Return the inverse variances of the training set fluxes. """
        return self._training_set_ivar


    @property
    def vectorizer(self):
        """ Return the vectorizer for this model. """
        return self._vectorizer


    @property
    def design_matrix(self):
        """ Return the design matrix for this model. """
        return self._design_matrix


    @property
    def theta(self):
        """ Return the theta coefficients (spectral model derivatives). """
        return self._theta


    @property
    def s2(self):
        """ Return the intrinsic variance (s^2) for all pixels. """
        return self._s2


    # Model attributes that can be changed after initiation.
    @property
    def censors(self):
        """ Return the wavelength censor masks for the labels. """
        return self._censors


    @censors.setter
    def censors(self, censors):
        """
        Set label censoring masks for each pixel.

        :param censors:
            A dictionary-like object with label names as keys, and boolean arrays
            as values.
        """

        censors = {} if censors is None else censors
        if isinstance(censors, censoring.Censors):
            # Could be a censoring dictionary from a different model,
            # with different label names and pixels.
            # So let's just extract what we need.
            censors = censors.items()
            
        if isinstance(censors, dict):
            self._censors = censoring.Censors(
                self.vectorizer.label_names, self.training_set_flux.shape[1],
                censors)
            return None

        else:
            raise TypeError(
                "censors must be a dictionary or a censoring.Censors object")


    @property
    def dispersion(self):
        """ Return the dispersion points for all pixels. """
        return self._dispersion


    @dispersion.setter
    def dispersion(self, dispersion):
        """
        Set the dispersion values for all the pixels.

        :param dispersion:
            An array of the dispersion values.
        """

        dispersion = np.array(dispersion).flatten()
        if dispersion.size != self.training_set_flux.shape[1]:
            raise ValueError("dispersion provided does not match the number "
                             "of pixels per star ({0} != {1})".format(
                                dispersion.size, self.training_set_flux.shape[1]))

        if dispersion.dtype.kind not in "iuf":
            raise ValueError("dispersion values are not float-like")

        if not np.all(np.isfinite(dispersion)):
            raise ValueError("dispersion values must be finite")

        self._dispersion = dispersion
        return None


    @property
    def regularization(self):
        """ Return the strength of the L1 regularization for this model. """
        return self._regularization


    @regularization.setter
    def regularization(self, regularization):
        """
        Specify the strength of the regularization for the model, either as a
        single value for all pixels, or a different strength for each pixel.

        :param regularization:
            The L1-regularization strength for the model.
        """

        if regularization is None:
            self._regularization = None
            return None

        regularization = np.array(regularization).flatten()
        if regularization.size == 1:
            regularization = regularization[0]
            if 0 > regularization or not np.isfinite(regularization):
                raise ValueError("regularization must be positive and finite")

        elif regularization.size != self.training_set_flux.shape[1]:
            raise ValueError("regularization array must be of size `num_pixels`")

            if any(0 > regularization) \
            or not np.all(np.isfinite(regularization)):
                raise ValueError("regularization must be positive and finite")

        self._regularization = regularization
        return None


    # Functions and other properties.
    @property
    def is_trained(self):
        """ Return true or false for whether the model is trained. """
        return all(getattr(self, attr, None) is not None \
            for attr in self._trained_attributes)


    def reset(self):
        """ Clear any attributes that have been trained. """
        for attribute in self._trained_attributes:
            setattr(self, "_{}".format(attribute), None)
        return None


    def _pixel_access(self, array, index, default=None):
        """
        Safely access a (potentially per-pixel) attribute of the model.
        
        :param array:
            Either `None`, a float value, or an array the size of the dispersion
            array.

        :param index:
            The zero-indexed pixel to attempt to access.

        :param default: [optional]
            The default value to return if `array` is None.
        """

        if array is None:
            return default
        try:
            return array[index]
        except TypeError:
            return array


    def _verify_training_data(self, rho_warning=0.90):
        """
        Verify the training data for the appropriate shape and content.

        :param rho_warning: [optional]
            Maximum correlation value between labels before a warning is given.
        """
        if self.training_set_flux.shape != self.training_set_ivar.shape:
            raise ValueError("the training set flux and inverse variance arrays"
                             " for the labelled set must have the same shape")

        if len(self.training_set_labels) != self.training_set_flux.shape[0]:
            raise ValueError(
                "the first axes of the training set flux array should "
                "have the same shape as the nuber of rows in the labelled set"
                "(N_stars, N_pixels)")

        if not np.all(np.isfinite(self.training_set_labels)):
            raise ValueError("training set labels are not all finite")

        if not np.all(np.isfinite(self.training_set_flux)):
            raise ValueError("training set fluxes are not all finite")

        if not np.all(self.training_set_ivar >= 0) \
        or not np.all(np.isfinite(self.training_set_ivar)):
            raise ValueError("training set ivars are not all positive finite")

        if self.dispersion is not None:
            dispersion = np.atleast_1d(self.dispersion).flatten()
            if dispersion.size != self.training_set_flux.shape[1]:
                raise ValueError(
                    "mis-match between the number of dispersion points and "
                    "normalised flux values ({0} != {1})".format(
                        self.training_set_flux.shape[1], dispersion.size))

        # Look for very high correlation coefficients between labels, which
        # could make the training time very difficult.
        rho = np.corrcoef(self.training_set_labels.T)

        # Set the diagonal indices to zero.
        K = rho.shape[0]
        rho[np.diag_indices(K)] = 0.0
        indices = np.argsort(rho.flatten())[::-1]

        for index in indices:
            x, y = (index % K, int(index / K)) 
            rho_xy = rho[x, y]
            if rho_xy >= rho_warning: 
                if x > y: # One warning per correlated label pair.
                    logger.warn("Labels '{X}' and '{Y}' are highly correlated ("\
                        "rho = {rho_xy:.2}). This may cause very slow training "\
                        "times. Are both labels needed?".format(
                            X=self.vectorizer.label_names[x],
                            Y=self.vectorizer.label_names[y],
                            rho_xy=rho_xy))
            else:
                break
        return None


    def in_convex_hull(self, labels):
        """
        Return whether the provided labels are inside a complex hull constructed
        from the labelled set.

        :param labels:
            A `NxK` array of `N` sets of `K` labels, where `K` is the number of
            labels that make up the vectorizer.

        :returns:
            A boolean array as to whether the points are in the complex hull of
            the labelled set.
        """

        labels = np.atleast_2d(labels)
        if labels.shape[1] != self.training_set_labels.shape[1]:
            raise ValueError("expected {} labels; got {}".format(
                self.training_set_labels.shape[1], labels.shape[1]))

        hull = Delaunay(self.training_set_labels)
        return hull.find_simplex(labels) >= 0


    def write(self, path, include_training_data=False, overwrite=False,
        protocol=-1):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param path:
            The path to save the model to.

        :param include_training_data: [optional]
            Save the labelled set, normalised flux and inverse variance used to
            train the model.

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.

        :param protocol: [optional]
            The Python pickling protocol to employ. Use 2 for compatibility with
            previous Python releases, -1 for performance.
        """

        raise IsItNeeded


    @classmethod
    def read(cls, path, **kwargs):
        """
        Load a saved model from disk.

        :param path:
            The path where to load the model from.
        """

        raise IsItNeeded
