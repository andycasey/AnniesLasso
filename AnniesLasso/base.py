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
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from six import string_types
from sys import version_info
from scipy.spatial import Delaunay

from .vectorizer import BaseVectorizer
from . import (censoring, utils, __version__)

logger = logging.getLogger(__name__)



class BaseCannonModel(object):
    """
    A base Cannon model object that implements data validation, properties,
    and convenience functions.
    """

    _data_attributes = \
        ("training_set_labels", "training_set_flux", "training_set_ivar")

    # Descriptive attributes are needed to train *and* test the model.
    _descriptive_attributes = \
        ("vectorizer", "censors", "regularization", "dispersion")

    # Trained attributes are set only at training time.
    _trained_attributes = ("theta", "s2")
    
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

        # Save the vectorizer.
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError(
                "vectorizer must be a sub-class of vectorizer.BaseVectorizer")
        
        self._vectorizer = vectorizer
        
        if training_set_flux is None and training_set_ivar is None:

            # Must be reading in a model that does not have the training set
            # spectra saved.
            self._training_set_flux = None
            self._training_set_ivar = None
            self._training_set_labels = training_set_labels

        else:
            self._training_set_flux = np.atleast_2d(training_set_flux)
            self._training_set_ivar = np.atleast_2d(training_set_ivar)
            self._training_set_labels = np.array(
                [training_set_labels[ln] for ln in vectorizer.label_names]).T
            
            # Check that the flux and ivar are valid.
            self._verify_training_data(**kwargs)

        # Set regularization, censoring, dispersion.
        self.regularization = regularization
        self.censors = censors
        self.dispersion = dispersion

        # Set useful private attributes.
        __scale_labels_function = kwargs.get("__scale_labels_function", 
            lambda l: np.ptp(np.percentile(l, [25.5, 97.5], axis=0, axis=0)))
        __fiducial_labels_function = kwargs.get("__fiducial_labels_function",
            lambda l: np.percentile(l, 50, axis=0))

        self._scales = __scale_labels_function(self.training_set_labels)
        self._fiducials = __fiducial_labels_function(self.training_set_labels)
        self._design_matrix = vectorizer(
            (self.training_set_labels - self._fiducials)/self._scales).T

        self.reset()

        return None

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


    # Model attributes that cannot (well, should not) be changed.


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
            
            # But more likely: we are loading a model from disk.
            self._censors = censors

        elif isinstance(censors, dict):
            self._censors = censoring.Censors(
                self.vectorizer.label_names, self.training_set_flux.shape[1],
                censors)

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
        if self.training_set_flux is not None \
        and dispersion.size != self.training_set_flux.shape[1]:
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


    # Convenient functions and properties.


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


    def write(self, path, include_training_set_spectra=False, overwrite=False,
        protocol=-1):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param path:
            The path to save the model to.

        :param include_training_set_spectra: [optional]
            Save the labelled set, normalised flux and inverse variance used to
            train the model.

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.

        :param protocol: [optional]
            The Python pickling protocol to employ. Use 2 for compatibility with
            previous Python releases, -1 for performance.
        """

        if os.path.exists(path) and not overwrite:
            raise IOError("path already exists: {0}".format(path))

        attributes = list(self._descriptive_attributes) \
                   + list(self._trained_attributes) \
                   + list(self._data_attributes)

        if "metadata" in attributes:
            logger.warn("'metadata' is a protected attribute. Ignoring.")
            attributes.remote("metadata")

        # Store up all the trained attributes and a hash of the training set.
        state = {}
        for attribute in attributes:

            value = getattr(self, attribute)

            try:
                # If it's a vectorizer or censoring dict, etc, get the state.
                value = value.__getstate__()
            except:
                None

            state[attribute] = value

        # Create a metadata dictionary.
        state["metadata"] = dict(
            version=__version__,
            model_class=type(self).__name__,
            modified=str(datetime.now()),
            data_attributes=self._data_attributes,
            descriptive_attributes=self._descriptive_attributes,
            trained_attributes=self._trained_attributes,
            training_set_hash=utils.short_hash(
                getattr(self, attr) for attr in self._data_attributes),
        )

        if not include_training_set_spectra:
            state.pop("training_set_flux")
            state.pop("training_set_ivar")

        elif not self.is_trained:
            logger.warn("The training set spectra won't be saved, and this model"\
                        "is not already trained. The saved model will not be "\
                        "able to be trained when loaded!")

        with open(path, "wb") as fp:
            pickle.dump(state, fp, protocol) 
        return None


    @classmethod
    def read(cls, path, **kwargs):
        """
        Read a saved model from disk.

        :param path:
            The path where to load the model from.
        """

        encodings = ("utf-8", "latin-1")
        for encoding in encodings:
            kwds = {"encoding": encoding} if version_info[0] >= 3 else {}
            try:
                with open(path, "rb") as fp:        
                    state = pickle.load(fp, **kwds)

            except UnicodeDecodeError:
                if encoding == encodings:
                    raise

        # Parse the state.
        metadata = state.get("metadata", {})
        version_saved = metadata.get("version", "0.1.0")
        if version_saved >= "0.2.0": # Refactor'd.

            init_attributes = list(metadata["data_attributes"]) \
                            + list(metadata["descriptive_attributes"])

            kwds = dict([(a, state.get(a, None)) for a in init_attributes])

            # Initiate the vectorizer.
            vectorizer_class, vectorizer_kwds = kwds["vectorizer"]
            klass = getattr(vectorizer, vectorizer_class)
            kwds["vectorizer"] = klass(**vectorizer_kwds)

            # Initiate the censors.
            kwds["censors"] = censoring.Censors(**kwds["censors"])

            model = cls(**kwds)

            # Set training attributes.
            for attr in metadata["trained_attributes"]:
                setattr(model, "_{}".format(attr), state.get(attr, None))

            return model
            
        else:
            raise NotImplementedError(
                "Cannot auto-convert old model files yet; "
                "contact Andy Casey <andrew.casey@monash.edu> if you need this")