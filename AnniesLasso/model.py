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
from datetime import datetime
from os import path
from six.moves import cPickle as pickle

from .interruptible_pool import InterruptiblePool
from .vectorizer.base import BaseVectorizer
from . import (utils, __version__ as code_version)

logger = logging.getLogger(__name__)


def requires_training_wheels(method):
    """
    A decorator for model methods that require training before being run.
    """

    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model needs training first")
        return method(model, *args, **kwargs)
    return wrapper


def requires_model_description(method):
    """
    A decorator for model methods that require a full model description.
    (That is, none of the _descriptive_attributes are None)
    """

    def wrapper(model, *args, **kwargs):
        for descriptive_attribute in model._descriptive_attributes:
            if getattr(model, descriptive_attribute) is None:
                raise TypeError("the model requires a {} term".format(
                    descriptive_attribute.lstrip("_")))
        return method(model, *args, **kwargs)
    return wrapper


class BaseCannonModel(object):
    """
    An abstract Cannon model object that implements convenience functions.

    :param labels:
        A table with columns as labels, and stars as rows.

    :type labels:
        :class:`~astropy.table.Table` or numpy structured array

    :param fluxes:
        An array of fluxes for stars in the training set, given as shape
        `(num_stars, num_pixels)`. The `num_stars` should match the number of
        rows in `labels`.

    :type fluxes:
        :class:`np.ndarray`

    :param flux_uncertainties:
        An array of 1-sigma flux uncertainties for stars in the training set,
        The shape of the `flux_uncertainties` should match `fluxes`. 

    :type flux_uncertainties:
        :class:`np.ndarray`

    :param dispersion: [optional]
        The dispersion values corresponding to the given pixels. If provided, 
        this should have length `num_pixels`.
    """

    _descriptive_attributes = ["_vectorizer"]
    _trained_attributes = ["_scatter", "_coefficients"]
    _data_attributes = \
        ["training_labels", "training_fluxes", "training_flux_uncertainties"]
    
    
    def __init__(self, labels, fluxes, flux_uncertainties, dispersion=None,
        threads=1, pool=None):

        self._training_labels = labels
        self._training_fluxes = np.atleast_2d(fluxes)
        self._training_flux_uncertainties = np.atleast_2d(flux_uncertainties)
        self._dispersion = np.arange(fluxes.shape[1], dtype=int) \
            if dispersion is None else dispersion
        
        # Initialise descriptive attributes for the model and verify the data.
        for attribute in self._descriptive_attributes:
            setattr(self, attribute, None)
        
        self._verify_training_data()

        self.reset()
        self.threads = threads
        self.pool = pool or InterruptiblePool(threads) if threads > 1 else None


    def __str__(self):
        return "<{module}.{name} {trained}using a training set of {N} stars "\
               "with {K} available labels and {M} pixels each>".format(
                    module=self.__module__,
                    name=type(self).__name__,
                    trained="trained " if self.is_trained else "",
                    N=len(self.training_labels),
                    K=len(self.labels_available),
                    M=len(self.dispersion))


    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(
            self.__module__, type(self).__name__, hex(id(self)))


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
        """
        try:
            len(dispersion)
        except TypeError:
            raise TypeError("dispersion provided must be an array or list-like")

        if len(dispersion) != self.training_fluxes.shape[1]:
            raise ValueError("dispersion provided does not match the number "
                             "of pixels per star ({0} != {1})".format(
                                len(dispersion), self.training_fluxes.shape[1]))

        dispersion = np.array(dispersion)
        if dispersion.dtype.kind not in "iuf":
            raise ValueError("dispersion values are not float-like")

        if not np.all(np.isfinite(dispersion)):
            raise ValueError("dispersion values must be finite")

        self._dispersion = dispersion
        return None


    @property
    def training_labels(self):
        return self._training_labels


    @property
    def training_fluxes(self):
        return self._training_fluxes


    @property
    def training_flux_uncertainties(self):
        return self._training_flux_uncertainties


    def _verify_training_data(self):
        """
        Verify the training data for the appropriate shape and content.
        """
        if self.training_fluxes.shape != self.training_flux_uncertainties.shape:
            raise ValueError(
                "the training flux and uncertainty arrays should "
                "have the same shape")

        if len(self.training_labels) == 0 \
        or self.training_labels.dtype.names is None:
            raise ValueError("no named labels provided for the training set")

        if len(self.training_labels) != self.training_fluxes.shape[0]:
            raise ValueError(
                "the first axes of the training flux array should "
                "have the same shape as the nuber of rows in the label table "
                "(N_stars, N_pixels)")

        if self.dispersion is not None:
            dispersion = np.atleast_1d(self.dispersion).flatten()
            if dispersion.size != self.training_fluxes.shape[1]:
                raise ValueError(
                    "mis-match between the number of wavelength "
                    "points ({0}) and flux values ({1})".format(
                        self.training_fluxes.shape[1], dispersion.size))
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
        if not isinstance(vectorizer, BaseVectorizer):
            raise TypeError("vectorizer must be "
                            "a sub-class of vectorizers.BaseVectorizer")
        self._vectorizer = vectorizer
        self.reset() # Any trained attributes must be reset.
        return None


    # Trained attributes that subclasses are likely to use.
    @property
    def coefficients(self):
        return self._coefficients


    @coefficients.setter
    def coefficients(self, coefficients):
        """
        Set the label vector coefficients for each pixel. This assumes a
        'standard' model where the label vector is common to all pixels.

        :param coefficients:
            A 2-d coefficients array of shape `(N_pixels, N_vectorizer_terms)`.
        """

        if coefficients is None:
            self._coefficients = None
            return None

        # Some sanity checks.
        coefficients = np.atleast_2d(coefficients)
        if len(coefficients.shape) > 2:
            raise ValueError("coefficients must be a 2D array")

        P, Q = coefficients.shape
        if P != len(self.dispersion):
            raise ValueError("axis 0 of coefficients array does not match the "
                             "number of pixels ({0} != {1})".format(
                                P, len(self.dispersion)))

        if Q != 1 + len(self.vectorizer.terms):
            raise ValueError("axis 1 of coefficients array does not match the "
                             "number of label vector terms ({0} != {1})".format(
                                Q, 1 + len(self.vectorizer.terms)))

        if np.any(np.all(~np.isfinite(coefficients), axis=0)):
            logger.warning("At least one vectorizer term has a non-finite "
                           "coefficient at all pixels")

        self._coefficients = coefficients
        return None


    @property
    def scatter(self):
        return self._scatter


    @scatter.setter
    def scatter(self, scatter):
        """
        Set the scatter values for each pixel.

        :param scatter:
            A 1-D array of scatter terms.
        """

        if scatter is None:
            self._scatter = None
            return None
        
        # Some sanity checks..
        scatter = np.array(scatter).flatten()
        if scatter.size != len(self.dispersion):
            raise ValueError("number of scatter values does not match "
                             "the number of pixels ({0} != {1})".format(
                                scatter.size, len(self.dispersion)))

        if np.any(scatter < 0):
            raise ValueError("scatter terms must be positive")

        if np.std(scatter) == 0:
            logger.warning("All pixels show the same level of variance!"
                           " (Something probably went very, very wrong)")
        self._scatter = scatter
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


    # I/O
    @requires_training_wheels
    def save(self, filename, include_training_data=False, overwrite=False):
        """
        Serialise the trained model and save it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param filename:
            The path to save the model to.

        :param include_training_data: [optional]
            Save the training data (labels, fluxes, uncertainties) used to train
            the model.

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.
        """

        if path.exists(filename) and not overwrite:
            raise IOError("filename already exists: {0}".format(filename))

        attributes = list(self._descriptive_attributes) \
            + list(self._trained_attributes) \
            + list(self._data_attributes)
        if "metadata" in attributes:
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
            pickle.dump(contents, fp, -1)

        return None


    def load(self, filename, verify_training_data=False):
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
            contents = pickle.load(fp)

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

        return None


    @property
    @requires_model_description
    def design_matrix(self):
        """
        Return the design matrix for all pixels.
        """
        design_matrix = self.vectorizer(np.vstack([
            self.training_labels[label] for label in self.vectorizer.labels]).T)

        if not np.all(np.isfinite(design_matrix)):
            logger.warn("Non-finite values in the design matrix!")
        return design_matrix


    # Residuals in labels in the training data set.
    @requires_training_wheels
    def get_training_label_residuals(self):
        """
        Return the residuals (model - training) between the parameters that the
        model returns for each star, and the training set value.
        """
        
        predicted_labels = self.fit(self.training_fluxes,
            self.training_flux_uncertainties, full_output=False)

        return predicted_labels - self.labels_array


    @requires_model_description
    def cross_validate(self, pre_train=None, **kwargs):
        """
        Perform leave-one-out cross-validation on the training set.
        """
        
        inferred = np.nan * np.ones_like(self.labels_array)
        N_training_set, N_labels = inferred.shape
        N_stop_at = kwargs.pop("N", N_training_set)

        debug = kwargs.pop("debug", False)
        
        kwds = { "threads": self.threads }
        kwds.update(kwargs)

        for i in range(N_training_set):
            
            training_set = np.ones(N_training_set, dtype=bool)
            training_set[i] = False

            # Create a clean model to use so we don't overwrite self.
            model = self.__class__(
                self.training_labels[training_set],
                self.training_fluxes[training_set],
                self.training_flux_uncertainties[training_set],
                **kwds)

            # Initialise and run any pre-training function.
            for _attribute in self._descriptive_attributes:
                setattr(model, _attribute[1:], getattr(self, _attribute[1:]))

            if pre_train is not None:
                pre_train(self, model)

            # Train and solve.
            model.train()

            try:
                inferred[i, :] = model.fit(self.training_fluxes[i],
                    self.training_flux_uncertainties[i], full_output=False)

            except:
                logger.exception("Exception during cross-validation on object "
                                 "with index {0}:".format(i))
                if debug: raise

            if i == N_stop_at + 1:
                break

        return inferred[:N_stop_at, :]


    @requires_training_wheels
    def define_continuum_mask(self, baseline_flux=None, tolerances=None,
        percentiles=None, absolute_percentiles=None):
        """
        Define a continuum mask based on constraints on the baseline flux values
        and the percentiles or absolute percentiles of theta coefficients. The
        resulting continuum mask is taken for whatever pixels meet all the given
        constraints.

        :param baseline_flux: [optional]
            The `(lower, upper`) range of acceptable baseline flux values to be 
            considered as continuum.

        :param percentiles: [optional]
            A dictionary containing the label vector description as keys and
            acceptable percentile ranges `(lower, upper)` for each corresponding
            label vector term.

        :param absolute_percentiles: [optional]
            The same as `percentiles`, except these are calculated on the
            absolute values of the model coefficients.
        """

        mask = np.ones_like(self.dispersion, dtype=bool)
        if baseline_flux is not None:
            if len(baseline_flux) != 2:
                raise ValueError("baseline flux constraints must be given as "
                                 "(lower, upper)")
            mask *= (max(baseline_flux) >= self.coefficients[:, 0]) \
                  * (self.coefficients[:, 0] >= min(baseline_flux))

        for term, constraints in (tolerances or {}).items():
            if len(constraints) != 2:
                raise ValueError("{} tolerance must be given as (lower, upper)"\
                    .format(term))

            p_term = utils.parse_label_vector(term)[0]
            if p_term not in self.label_vector:
                logger.warn("Term {0} ({1}) is not in the label vector, "
                            "and is therefore being ignored".format(
                                term, p_term))
                continue

            a = self.coefficients[:, 1 + self.label_vector.index(p_term)]
            mask *= (max(constraints) >= a) * (a >= min(constraints))

        for qs, use_abs in zip([percentiles, absolute_percentiles], [0, 1]):
            if qs is None: continue

            for term, constraints in qs.items():
                if len(constraints) != 2:
                    raise ValueError("{} constraints must be given as "
                                     "(lower, upper)".format(term))

                p_term = utils.parse_label_vector(term)[0]
                if p_term not in self.label_vector:
                    logger.warn("Term {0} ({1}) is not in the label vector, "
                                "and is therefore being ignored".format(
                                    term, p_term))
                    continue

                a = self.coefficients[:, 1 + self.label_vector.index(p_term)]
                if use_abs: a = np.abs(a)

                p = np.percentile(a, constraints)
                mask *= (max(p) >= a) * (a >= min(p))

        return mask

