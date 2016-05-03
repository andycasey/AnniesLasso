#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An abstract model class for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = [
    "BaseCannonModel", "requires_training_wheels", "requires_model_description"]


import logging
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from os import path
from six.moves import cPickle as pickle
from six import string_types

from .vectorizer.base import BaseVectorizer
from . import (censoring, utils, __version__ as code_version)

logger = logging.getLogger(__name__)


def requires_training_wheels(method):
    """
    A decorator for model methods that require training before being run.

    :param method:
        A method belonging to a sub-class of BaseCannonModel.
    """
    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model needs training first")
        return method(model, *args, **kwargs)
    return wrapper


def requires_model_description(*args):
    """
    A decorator for model methods that require some part of a model description.

    :param args: [optional]
        If specified, only these descriptors will be required. If set as None,
        then all model descriptors will be required.
    """

    def decorator(method):
        """
        A decorator for model methods that require a full model description.

        :param method:
            A method belonging to a sub-class of BaseCannonModel.
        """
    
        def wrapper(model, *args, **kwargs):
            for attr in descriptors or model._descriptive_attributes:
                if getattr(model, attr) is None:
                    raise TypeError("the model requires a {} term".format(
                        attr.lstrip("_")))
            return method(model, *args, **kwargs)
        return wrapper

    # This decorator can have optional descriptors.
    descriptors = None if len(args) == 1 and callable(args[0]) else args[1:]
    if descriptors is None:
        return decorator(args[0])
    return decorator



class BaseCannonModel(object):
    """
    An abstract Cannon model object that implements convenience functions.

    :param labelled_set:
        A set of labelled objects. The most common input form is a table with
        columns as labels, and stars/objects as rows.

    :type labelled_set:
        A numpy structured array.

    :param normalized_flux:
        An array of normalised fluxes for stars in the labelled set, given as
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

    :param copy: [optional]
        Make deep copies of the `labelled_set`, `normalized_flux`,
        `normalized_ivar`, and the `dispersion` (when applicable).
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


    def _init_data_attributes(self, labelled_set, normalized_flux,
        normalized_ivar, mode="c", dtype=float):
        """
        Initialize the labelled set data attributes.

        :param labelled_set:
            A table of labelled stars.

        :param normalized_flux:
            An array of normalized fluxes for stars in the labelled set, or a
            file path to a float-like memory-mapped array.

        :param normalized_ivar:
            An array of normalized inverse variances for stars in the labelled 
            set, or a file path to a float-like memory-mapped array.
        """

        is_path = lambda p: isinstance(p, string_types) and path.exists(p)
        if is_path(normalized_flux):
            normalized_flux = np.memmap(normalized_flux, mode=mode, dtype=dtype)
            normalized_flux = normalized_flux.reshape((len(labelled_set), -1))

        if is_path(normalized_ivar):
            normalized_ivar = np.memmap(normalized_ivar, mode=mode, dtype=dtype)
            normalized_ivar = normalized_ivar.reshape((len(labelled_set), -1))
            
        self._labelled_set = labelled_set
        self._normalized_flux = np.atleast_2d(normalized_flux)
        self._normalized_ivar = np.atleast_2d(normalized_ivar)
        return None


    def __str__(self):
        return "<{module}.{name} {trained}using a training set of {N} stars "\
               "with {M} pixels each>".format(module=self.__module__,
                    name=type(self).__name__,
                    trained="trained " if self.is_trained else "",
                    N=len(self.labelled_set), M=len(self.dispersion))


    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(self.__module__, 
            type(self).__name__, hex(id(self)))


    def copy(self):
        """
        Create a (serial; unparallelized) copy of the current model.
        """

        model = self.__class__(self.labelled_set.copy(), 
            self._normalized_flux.copy(), self._normalized_ivar.copy(),
            dispersion=self.dispersion.copy())
        attributes = ["_metadata"] + \
            self._descriptive_attributes + self._trained_attributes
        for attribute in attributes:
            setattr(model, attribute, deepcopy(getattr(self, attribute, None)))
        return model


    def reset(self):
        """
        Clear any attributes that have been trained upon.
        """
        for attribute in self._trained_attributes:
            setattr(self, attribute, None)
        return None


    # Attributes related to the training set.
    @property
    def dispersion(self):
        """
        Return the dispersion points for all pixels.
        """
        return self._dispersion


    @dispersion.setter
    def dispersion(self, dispersion):
        """
        Set the dispersion values for all the pixels.

        :param dispersion:
            An array of the dispersion values.
        """
        try:
            len(dispersion)
        except TypeError:
            raise TypeError("dispersion provided must be an array or list-like")

        if len(dispersion) != self.normalized_flux.shape[1]:
            raise ValueError("dispersion provided does not match the number "
                             "of pixels per star ({0} != {1})".format(
                                len(dispersion), self.normalized_flux.shape[1]))

        dispersion = np.array(dispersion)
        if dispersion.dtype.kind not in "iuf":
            raise ValueError("dispersion values are not float-like")

        if not np.all(np.isfinite(dispersion)):
            raise ValueError("dispersion values must be finite")

        self._dispersion = dispersion
        return None


    @property
    def labelled_set(self):
        """
        Return the table of stars (as rows) in the labelled set.
        """
        return self._labelled_set


    @property
    def normalized_flux(self):
        """
        Return the normalized fluxes of pixels for stars in the labelled set.
        """ 
        return self._normalized_flux


    @property
    def normalized_ivar(self):
        """
        Return the normalized inverse variances of pixels for stars in the
        labelled set.
        """
        return self._normalized_ivar


    def _verify_training_data(self):
        """
        Verify the training data for the appropriate shape and content.
        """
        if self.normalized_flux.shape != self.normalized_ivar.shape:
            raise ValueError("the normalized flux and inverse variance arrays "
                             "for the labelled set must have the same shape")

        if len(self.labelled_set) == 0 \
        or self.labelled_set.dtype.names is None:
            raise ValueError("no named labels provided in the labelled set")

        if len(self.labelled_set) != self.normalized_flux.shape[0]:
            raise ValueError(
                "the first axes of the normalised flux array should "
                "have the same shape as the nuber of rows in the labelled set"
                "(N_stars, N_pixels)")

        if self.dispersion is not None:
            dispersion = np.atleast_1d(self.dispersion).flatten()
            if dispersion.size != self.normalized_flux.shape[1]:
                raise ValueError(
                    "mis-match between the number of dispersion points and "
                    "normalised flux values ({0} != {1})".format(
                        self.normalized_flux.shape[1], dispersion.size))
        return None


    @property
    def vectorizer(self):
        """
        Return the vectorizer for this Cannon model.
        """
        return self._vectorizer


    @vectorizer.setter
    def vectorizer(self, vectorizer):
        """
        Set the vectorizer for this Cannon model.
        """
        if vectorizer is None:
            self._vectorizer = None
            return None

        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("vectorizer must be "
                            "a sub-class of vectorizers.BaseVectorizer")
        self._vectorizer = vectorizer
        return None


    @property
    def censors(self):
        """
        Return the wavelength censor masks for the labels.
        """
        return self._censors


    @censors.setter
    def censors(self, censors):
        """
        Set the wavelength censors (per label) for the model.

        :param censors:
            A dictionary containing the labels as keys and masks as values. The
            masks can be in the form of a boolean array of size `N_pixels`,
            or a list of two-length tuples containing (start, end) regions to
            exclude.
        """

        if censors is not None:
            if self.vectorizer is None:
                # Wavelength censoring can't be set if we don't know the 
                # vectorizer names.
                raise TypeError("model requires a vectorizer to validate the "
                    "wavelength censors")

            if not isinstance(censors, (dict, censoring.CensorsDict)):
                raise TypeError("censors must be given as a dictionary")

        # Create a new dictionary, regardless of whether censors is None or not.
        self._censors = censoring.CensorsDict(self)

        if censors is not None:
            self._censors.update(censors)

        return None


    # Trained attributes that subclasses are likely to use.
    @property
    def theta(self):
        """
        Return the theta coefficients (spectral model derivatives).
        """
        return self._theta


    @theta.setter
    def theta(self, theta):
        """
        Set the theta coefficients for the vectorizer at each pixel.

        :param theta:
            A 2-d theta array of shape `(N_pixels, N_vectorizer_terms)`.
        """

        if theta is None:
            self._theta = None
            return None

        # Some sanity checks.
        theta = np.atleast_2d(theta)
        if len(theta.shape) > 2:
            raise ValueError("theta must be a 2D array")

        P, Q = theta.shape
        if P != len(self.dispersion):
            raise ValueError("axis 0 of theta array does not match the "
                             "number of pixels ({0} != {1})".format(
                                P, len(self.dispersion)))

        if Q != 1 + len(self.vectorizer.terms):
            raise ValueError("axis 1 of theta array does not match the "
                             "number of label vector terms ({0} != {1})".format(
                                Q, 1 + len(self.vectorizer.terms)))

        if np.any(np.all(~np.isfinite(theta), axis=0)):
            logger.warning("At least one vectorizer term has a non-finite "
                           "coefficient at all pixels")

        self._theta = theta
        return None


    @property
    def s2(self):
        """
        Return the intrinsic variance (s^2) for all pixels.
        """
        return self._s2


    @s2.setter
    def s2(self, s2):
        """
        Set the intrisic variance term for all pixels.

        :param s2:
            A 1-d array of `s^2` (intrinsic variance) values.
        """

        if s2 is None:
            self._s2 = None
            return None
        
        # Some sanity checks..
        s2 = np.array(s2).flatten()
        if s2.size == 1:
            s2 = np.ones_like(self.dispersion) * s2[0]
        elif s2.size != len(self.dispersion):
            raise ValueError("number of variance values does not match "
                             "the number of pixels ({0} != {1})".format(
                                s2.size, len(self.dispersion)))
        if np.any(s2 < 0):
            raise ValueError("the intrinsic variance terms must be positive")

        self._s2 = s2
        return None


    @property
    def is_trained(self):
        return all(getattr(self, attr, None) is not None \
            for attr in self._trained_attributes)


    def train(self, *args, **kwargs):
        raise NotImplementedError("The train method must be "
                                  "implemented by subclasses")


    def predict(self, *args, **kwargs):
        raise NotImplementedError("The predict method must be "
                                  "implemented by subclasses")


    def fit(self, *args, **kwargs):
        raise NotImplementedError("The fit method must be "
                                  "implemented by subclasses")


    def save(self, filename, include_training_data=False, overwrite=False,
        protocol=-1):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param filename:
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

        if path.exists(filename) and not overwrite:
            raise IOError("filename already exists: {0}".format(filename))

        attributes = list(self._descriptive_attributes) \
            + list(self._trained_attributes) \
            + list(self._data_attributes)
        if "metadata" in attributes or "_metadata" in attributes:
            raise ValueError("'metadata' is a protected attribute and cannot "
                             "be used in the _*_attributes in a class")

        # Store up all the trained attributes and a hash of the training set.
        contents = OrderedDict([
            (attr.lstrip("_"), getattr(self, attr)) for attr in \
            (self._descriptive_attributes + self._trained_attributes)])
        contents["training_set_hash"] = utils.short_hash(getattr(self, attr) \
            for attr in self._data_attributes)

        if include_training_data:
            contents.update([(attr.lstrip("_"), getattr(self, attr)) \
                for attr in self._data_attributes])

        contents["metadata"] = {
            "version": code_version,
            "metadata": getattr(self, "_metadata", {}),
            "model_name": type(self).__name__, 
            "modified": str(datetime.now()),
            "data_attributes": \
                [_.lstrip("_") for _ in self._data_attributes],
            "trained_attributes": \
                [_.lstrip("_") for _ in self._trained_attributes],
            "descriptive_attributes": \
                [_.lstrip("_") for _ in self._descriptive_attributes]
        }

        with open(filename, "wb") as fp:
            pickle.dump(contents, fp, protocol) 

        return None


    def load(self, filename, verify_training_data=False, **kwargs):
        """
        Load a saved model from disk.

        :param filename:
            The path where to load the model from.

        :param verify_training_data: [optional]
            If there is training data in the saved model, verify its contents.
            Otherwise if no training data is saved, verify that the data used
            to train the model is the same data provided when this model was
            instantiated.
        """
        with open(filename, "rb") as fp:
            contents = pickle.load(fp, **kwargs)

        assert contents["metadata"]["model_name"] == type(self).__name__

        # If data exists, deal with that first.
        has_data = (contents["metadata"]["data_attributes"][0] in contents)
        if has_data:

            if verify_training_data:
                data_hash = utils.short_hash(contents[attr] \
                    for attr in contents["metadata"]["data_attributes"])
                if contents["training_set_hash"] is not None \
                and data_hash != contents["training_set_hash"]:
                    raise ValueError("expected hash for the training data is "
                                     "different to the actual data hash "
                                     "({0} != {1})".format(
                                        contents["training_set_hash"],
                                        data_hash))

            # Set the data attributes.
            for attribute in contents["metadata"]["data_attributes"]:
                if attribute in contents:
                    setattr(self, "_{}".format(attribute), contents[attribute])

        # Set descriptive and trained attributes.
        self.reset()
        for attribute in contents["metadata"]["descriptive_attributes"]:
            setattr(self, "_{}".format(attribute), contents[attribute])
        for attribute in contents["metadata"]["trained_attributes"]:
            setattr(self, "_{}".format(attribute), contents[attribute])

        # And update the metadata.
        setattr(self, "_metadata", contents["metadata"].get("metadata", {}))
        return self


    @property
    @requires_model_description("vectorizer")
    def design_matrix(self):
        """
        Return the design matrix for all pixels.
        """
        matrix = self.vectorizer(np.vstack([self.labelled_set[label_name] \
            for label_name in self.vectorizer.label_names]).T)

        if not np.all(np.isfinite(matrix)):
            raise ValueError("non-finite values in the design matrix!")

        return matrix


    @property
    @requires_model_description("vectorizer", "censors")
    def censored_vectorizer_terms(self):
        """
        Return a mask of which indices in the design matrix columns should be
        used for a given pixel. 
        """        

        # Parse all the terms once-off.
        mapper = {}
        pixel_masks = np.atleast_2d(list(map(list, self.censors.values())))
        for i, terms in enumerate(self.vectorizer.terms):
            for label_index, power in terms:
                # Let's map this directly to the censors that we actually have.
                try:
                    censor_index = list(self.censors.keys()).index(
                        self.vectorizer.label_names[label_index])

                except ValueError:
                    # Label name is not censored, so we don't care.
                    continue

                else:
                    # Initialize a list if necessary.
                    mapper.setdefault(censor_index, [])

                    # Note that we add +1 because the first term in the design
                    # matrix columns will actually be the pivot point.
                    mapper[censor_index].append(1 + i)

        # We already know the number of terms from i.
        mask = np.ones((self.dispersion.size, 2 + i), dtype=bool)
        for censor_index, pixel in zip(*np.where(pixel_masks)):
            mask[pixel, mapper[censor_index]] = False

        return mask


    @property
    def labels_array(self):
        """
        Return an array of all the label values in the labelled set which
        contribute to the design matrix.
        """
        return self.get_labels_array(self.labelled_set)


    @requires_model_description("vectorizer")
    def get_labels_array(self, labelled_set):
        return np.vstack([labelled_set[label_name] \
            for label_name in self.vectorizer.label_names]).T


    @requires_training_wheels
    def fit_labelled_set(self):
        """
        Return the predicted labels of the stars in the labelled set.
        """
        return self.fit(self.normalized_flux, self.normalized_ivar,
            full_output=False)


    @requires_model_description
    def loo_cross_validation(self, pre_train=None, **kwargs):
        """
        Perform leave-one-out cross-validation on the labelled set.
        """
        
        inferred = np.nan * np.ones_like(self.labels_array)
        N_training_set, N_labels = inferred.shape
        N_stop_at = kwargs.pop("N", N_training_set)

        debug = kwargs.pop("debug", False)
        
        kwds = { "threads": self._metadata.get("threads", 1) }
        kwds.update(kwargs)

        logger.debug("LOO-CV options: {0} {1} {2} {3} {4}".format(
            N_training_set, N_labels, N_stop_at, debug, kwds))

        for i in range(N_training_set):
            
            training_set = np.ones(N_training_set, dtype=bool)
            training_set[i] = False

            # Create a clean model to use so we don't overwrite self.
            model = self.__class__(
                self.labelled_set[training_set],
                self.normalized_flux[training_set],
                self.normalized_ivar[training_set],
                **kwds)

            # Initialise and run any pre-training function.
            for _attribute in self._descriptive_attributes:
                setattr(model, _attribute[1:], getattr(self, _attribute[1:]))

            if pre_train is not None:
                pre_train(self, model)

            # Train and solve.
            model.train()

            try:
                inferred[i, :] = model.fit(self.normalized_flux[i],
                    self.normalized_ivar[i], full_output=False)

            except:
                logger.exception("Exception during cross-validation on object "
                                 "with index {0}:".format(i))
                if debug: raise

            if i == N_stop_at + 1:
                break

        return inferred[:N_stop_at, :]


def _chi_sq(theta, design_matrix, data, inv_var, axis=None):
    """
    Calculate the chi-squared difference between the spectral model and data.
    """
    residuals = np.dot(theta, design_matrix.T) - data

    return np.sum(inv_var * residuals**2, axis=axis), \
        2.0 * inv_var * residuals * design_matrix.T


