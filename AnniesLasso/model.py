#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An abstract model class for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseCannonModel", "requires_training_wheels"]

import logging
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from os import path
from six.moves import cPickle as pickle

from . import utils

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

    __data_attributes = []
    __trained_attributes = []
    __forbidden_label_characters = None
    
    def __init__(self, labels, fluxes, flux_uncertainties, dispersion=None,
        threads=1, pool=None, live_dangerously=False):

        self._training_labels = labels
        self._training_fluxes = np.atleast_2d(fluxes)
        self._training_flux_uncertainties = np.atleast_2d(flux_uncertainties)
        
        self._pivot = False
        self._trained = False
        self._dispersion = np.arange(fluxes.shape[1], dtype=int) \
            if dispersion is None else dispersion
        
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

        attrs = [] + list(self.__trained_attributes) + \
            ["training_label_residuals", "_lowest_order_label_indices"]

        for attr in self.__trained_attributes:
            try:
                delattr(self, "_{}".format(attr))
            except AttributeError:
                continue

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
        if self.__forbidden_label_characters is None:
            return True

        for label in self.training_labels.dtype.names:
            if any(char in label for char in self.__forbidden_label_characters):
                raise ValueError(
                    "forbidden character '{char}' is in potential "
                    "label '{label}' - you can disable this verification by "
                    "enabling live_dangerously".format(char=char, label=label))
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
        return getattr(self, "_label_vector", None)


    @label_vector.setter
    def label_vector(self, label_vector_description):
        """
        Set a label vector.

        :param label_vector_description:
            A structured or human-readable version of the label vector
            description.
        """

        label_vector = utils.parse_label_vector(label_vector_description)

        # Need to actually verify that the parameters listed in the label vector
        # are actually present in the training labels.
        missing = \
            set(self._get_labels(label_vector)).difference(self.labels_available)
        if missing:
            raise ValueError("the following labels parsed from the label vector "
                             "description are missing in the training set of "
                             "labels: {0}".format(", ".join(missing)))

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


    @property
    def _lowest_order_label_indices(self):
        try:
            return self.__lowest_order_label_indices
        except AttributeError:
            self.__lowest_order_label_indices = \
                self._get_lowest_order_label_indices()
        return self.__lowest_order_label_indices


    def _get_lowest_order_label_indices(self):
        """
        Get the indices for the lowest power label terms in the label vector.
        """
        indices = OrderedDict()
        for i, term in enumerate(self.label_vector):
            if len(term) > 1: continue
            label, order = term[0]
            if order < indices.get(label, np.inf):
                indices[label] = i

        return [indices.get(label, None) for label in self.labels]


    # Trained attributes that subclasses are likely to use.
    @property
    def coefficients(self):
        return getattr(self, "_coefficients", None)


    @coefficients.setter
    def coefficients(self, coefficients):
        """
        Set the label vector coefficients for each pixel. This assumes a
        'standard' model where the label vector is common to all pixels.

        :param coefficients:
            A 2-D array of coefficients of shape 
            (`N_pixels`, `N_label_vector_terms`).
        """

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
        return getattr(self, "_scatter", None)


    @scatter.setter
    def scatter(self, scatter):
        """
        Set the scatter values for each pixel.

        :param scatter:
            A 1-D array of scatter terms.
        """
        
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


    def solve_labels(self, *args, **kwargs):
        raise NotImplementedError("The solve_labels method must be "
                                  "implemented by subclasses")


    # I/O
    @requires_training_wheels
    def write(self, filename, with_training_data=False, overwrite=False):
        """
        Serialise the trained model and write it to disk. This will save all
        relevant training attributes, and optionally, the training data.

        :param filename:
            The path to save the model to.

        :param with_training_data: [optional]
            Save the training data (labels, fluxes, uncertainties) used to train
            the model.

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.
        """

        if not self.is_trained:
            raise TypeError("the model needs training first")

        contents = [getattr(self, attr) for attr in self.__training_attributes]
        contents += [utils.short_hash(
            getattr(self, attr) for attr in self.__data_attributes)]

        if with_training_data:
            contents.extend(
                [getattr(self, attr) for attr in self.__data_attributes])

        if path.exists(filename) and not overwrite:
            raise IOError("filename already exists: {0}".format(filename))

        with open(filename, "w") as fp:
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

        with open(filename, "r") as fp:
            contents = pickle.load(fp)

        # Contents includes trained attributes, a data hash, and optionally the
        # training data.
        trained_contents = dict(zip(self.__training_attributes, contents))
        
        N = len(trained_contents)
        expected_data_hash = contents[N]

        if len(contents) > N + 1:
            data_contents = dict(zip(self.__data_attributes, contents[N + 1:]))
            if verify_training_data and expected_data_hash is not None:
                actual_data_hash = utils.short_hash(data_contents)
                if actual_data_hash != expected_data_hash:
                    raise ValueError(
                        "expected hash for the training data ({0}) "
                        "is different to the actual data hash ({1})".format(
                            expected_data_hash, actual_data_hash))

            # Set the data attributes.
            for k, v in data_contents.items():
                setattr(self, k, v)

        # Set the training attributes.
        self.reset()
        for k, v in trained_contents.items():
            setattr(self, k, v)

        self._trained = True
        return None


    # Properties and attribuets related to training, etc.
    @property
    @requires_label_vector
    def training_label_array(self):
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
            print("Non-finite labels identified in the label vector array!")
        return (lva, offsets)


    # Residuals in labels in the training data set.
    @property
    def training_label_residuals(self):
        """
        Label residuals for stars in the training set.
        """
        if not hasattr(self, "_training_label_residuals"):
            self._training_label_residuals = self._get_training_label_residuals()
        return self._training_label_residuals


    @requires_training_wheels
    def _get_training_label_residuals(self):
        """
        Return the residuals (model - training) between the parameters that the
        model returns for each star, and the training set value.
        """
        
        expected_labels = self.training_label_array
        optimised_labels = self.solve_labels(
            self.training_fluxes, self.training_flux_uncertainties,
            full_output=False)

        return optimised_labels - expected_labels

    # Put Cross-validation functions in here.
    @requires_training_wheels
    def cross_validate(self, *args, **kwargs):
        """
        Perform leave-one-out cross-validation on the training set.
        """
        
        expected = self.training_label_array
        inferred = np.nan * np.ones_like(expected)
        N_training_set, N_labels = inferred.shape

        debug = kwargs.pop("debug", False)
        pre_train = kwargs.pop("pre_train", None)

        kwds = { "threads": self.threads }
        kwds.update(kwargs)

        for i in range(N_training_set):
            
            use = np.zeros(N_training_set, dtype=bool)
            use[i] = True

            # Create a clean model to use so we don't overwrite self.
            model = self.__class__(
                self.labels[use],
                self.training_fluxes[use],
                self.training_flux_uncertainties[use],
                **kwds)

            # Initialise and run any pre-training function.
            model.label_vector = self.label_vector
            if pre_train is not None:
                pre_train(self, model)

            model.train()


            # Solve for the one object left out.
            try:
                inferred[i, :] = model.fit(self.training_fluxes[i],
                    self.training_flux_uncertainties[i], full_output=False)

            except:
                logger.exception("Exception during cross-validation on object "
                                 "with index {0}:".format(i))
                if debug: raise

        return inferred


    @requires_training_wheels
    def cross_validate_by_label(self, *args, **kwargs):
        raise NotImplementedError("not done yet")



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
