#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An abstract model class for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = \
    ["BaseCannonModel", "requires_training_wheels", "requires_label_vector"]

import logging
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from datetime import datetime
from os import path
from six.moves import cPickle as pickle

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


def requires_label_vector(method):
    """
    A decorator for model methods that require a label vector description.
    """

    def wrapper(model, *args, **kwargs):
        if model.label_vector is None:
            raise TypeError("the model requires a label vector description")
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

    :param live_dangerously: [optional]
        If enabled then no checks will be made on the label names, prohibiting
        the user to input human-readable forms of the label vector.
    """

    _descriptive_attributes = ["_label_vector"]
    _trained_attributes = []
    _data_attributes = []
    _forbidden_label_characters = "^*"
    
    def __init__(self, labels, fluxes, flux_uncertainties, dispersion=None,
        threads=1, pool=None, live_dangerously=False):

        self._training_labels = labels
        self._training_fluxes = np.atleast_2d(fluxes)
        self._training_flux_uncertainties = np.atleast_2d(flux_uncertainties)
        self._dispersion = np.arange(fluxes.shape[1], dtype=int) \
            if dispersion is None else dispersion
        
        for attribute in self._descriptive_attributes:
            setattr(self, attribute, None)
        
        # The training data must be checked, but users can live dangerously if
        # they think they can correctly specify the label vector description.
        self._verify_training_data()
        if not live_dangerously:
            self._verify_labels_available()

        self.reset()
        self.threads = threads
        self.pool = pool or mp.Pool(threads) if threads > 1 else None


    def reset(self):
        """
        Clear any attributes that have been trained upon.
        """

        self._trained = False
        for attribute in self._trained_attributes:
            setattr(self, attribute, None)
        
        return None


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


    # Attributes related to the training data.
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


    # Verifying the training data.
    def _verify_labels_available(self):
        """
        Verify the label names provided do not include forbidden characters.
        """
        if self._forbidden_label_characters is None:
            return True

        for label in self.training_labels.dtype.names:
            for character in self._forbidden_label_characters:
                if character in label:
                    raise ValueError(
                        "forbidden character '{char}' is in potential "
                        "label '{label}' - you can disable this verification "
                        "by enabling `live_dangerously`".format(
                            char=character, label=label))
        return None


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
    def is_trained(self):
        return self._trained


    # Attributes related to the labels and the label vector description.
    @property
    def labels_available(self):
        """
        All of the available labels for each star in the training set.
        """
        return self.training_labels.dtype.names


    @property
    def label_vector(self):
        """ The label vector for all pixels. """
        return self._label_vector


    @label_vector.setter
    def label_vector(self, label_vector_description):
        """
        Set a label vector.

        :param label_vector_description:
            A structured or human-readable version of the label vector
            description.
        """

        if label_vector_description is None:
            self._label_vector = None
            self.reset()
            return None

        label_vector = utils.parse_label_vector(label_vector_description)

        # Need to actually verify that the parameters listed in the label vector
        # are actually present in the training labels.
        missing = \
            set(self._get_labels(label_vector)).difference(self.labels_available)
        if missing:
            raise ValueError("the following labels parsed from the label "
                             "vector description are missing in the training "
                             "set of labels: {0}".format(", ".join(missing)))

        # If this is really a new label vector description,
        # then we are no longer trained.
        if not hasattr(self, "_label_vector") \
        or label_vector != self._label_vector:
            self._label_vector = label_vector
            self.reset()

        return None


    @property
    def human_readable_label_vector(self):
        """ Return a human-readable form of the label vector. """
        return utils.human_readable_label_vector(self.label_vector)


    @property
    def labels(self):
        """ The labels that contribute to the label vector. """
        return self._get_labels(self.label_vector)
    

    def _get_labels(self, label_vector):
        """
        Return the labels that contribute to the structured label vector
        provided.
        """
        return () if label_vector is None else \
            list(OrderedDict.fromkeys([label for term in label_vector \
                for label, power in term if power != 0]))


    def _get_lowest_order_label_indices(self):
        """
        Get the indices for the lowest power label terms in the label vector.
        """
        indices = OrderedDict()
        for i, term in enumerate(self.label_vector):
            if len(term) > 1: continue
            label, order = term[0]
            if order < indices.get(label, [None, np.inf])[-1]:
                indices[label] = (i, order)
        return [indices.get(label, [None])[0] for label in self.labels]


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
            A 2-D array of coefficients of shape 
            (`N_pixels`, `N_label_vector_terms`).
        """

        if coefficients is None:
            self._coefficients = None
            return None

        coefficients = np.atleast_2d(coefficients)
        if len(coefficients.shape) > 2:
            raise ValueError("coefficients must be a 2D array")

        P, Q = coefficients.shape
        if P != len(self.dispersion):
            raise ValueError("axis 0 of coefficients array does not match the "
                             "number of pixels ({0} != {1})".format(
                                P, len(self.dispersion)))
        if Q != 1 + len(self.label_vector):
            raise ValueError("axis 1 of coefficients array does not match the "
                             "number of label vector terms ({0} != {1})".format(
                                Q, 1 + len(self.label_vector)))
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
        
        scatter = np.array(scatter).flatten()
        if scatter.size != len(self.dispersion):
            raise ValueError("number of scatter values does not match "
                             "the number of pixels ({0} != {1})".format(
                                scatter.size, len(self.dispersion)))
        if np.any(scatter < 0):
            raise ValueError("scatter terms must be positive")
        self._scatter = scatter
        return None


    # Methods which must be implemented or updated by the subclasses.
    def pixel_label_vector(self, pixel_index):
        """ The label vector for a given pixel. """
        return self.label_vector


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
        self._trained = True

        return None


    # Properties and attribuets related to training, etc.
    @property
    @requires_label_vector
    def labels_array(self):
        """
        Return an array containing just the training labels, given the label
        vector.
        """
        return _build_label_vector_rows(
            [[(label, 1)] for label in self.labels], self.training_labels)[1:].T


    @property
    def label_vector_array(self):
        if not hasattr(self, "_label_vector_array"):
            self._label_vector_array, self.pivot_offsets \
                = self._get_label_vector_array()
        return self._label_vector_array


    @requires_label_vector
    def _get_label_vector_array(self):
        """
        Build the label vector array.
        """

        offsets = 0
        lva = _build_label_vector_rows(self.label_vector, self.training_labels)

        if not np.all(np.isfinite(lva)):
            logger.warn("Non-finite labels in the label vector array!")
        return (lva, offsets)


    # Residuals in labels in the training data set.
    @requires_training_wheels
    def get_training_label_residuals(self):
        """
        Return the residuals (model - training) between the parameters that the
        model returns for each star, and the training set value.
        """
        
        optimised_labels = self.fit(self.training_fluxes,
            self.training_flux_uncertainties, full_output=False)

        return optimised_labels - self.labels_array


    def _format_input_labels(self, args=None, **kwargs):
        """
        Format input labels either from a list or dictionary into a common form.
        """

        # We want labels in a dictionary.
        labels = kwargs if args is None else dict(zip(self.labels, args))
        return { k: (v if isinstance(v, (list, tuple, np.ndarray)) else [v]) \
            for k, v in labels.items() }


    # Put Cross-validation functions in here.
    @requires_label_vector
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


def _build_label_vector_rows(label_vector, training_labels):
    """
    Build a label vector row from a description of the label vector (as indices
    and orders to the power of) and the label values themselves.

    For example: if the first item of `labels` is `A`, and the label vector
    description is `A^3` then the first item of `label_vector` would be:

    `[[(0, 3)], ...`

    This indicates the first label item (index `0`) to the power `3`.

    :param label_vector:
        An `(index, order)` description of the label vector. 

    :param training_labels:
        The values of the corresponding training labels.

    :returns:
        The corresponding label vector row.
    """

    columns = [np.ones(len(training_labels), dtype=float)]
    for term in label_vector:
        column = 1.
        for label, order in term:
            column *= np.array(training_labels[label]).flatten()**order
        columns.append(column)

    try:
        return np.vstack(columns)

    except ValueError:
        columns[0] = np.ones(1)
        return np.vstack(columns)
