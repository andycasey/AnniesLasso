#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An abstract model class for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseCannonModel", "requires_training_wheels"]

import cPickle as pickle
import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from os import path

from . import utils


def requires_training_wheels(f):
    """
    A decorator for model functions that require training before being run.
    """

    def wrapper(model, *args, **kwargs):
        if not model.is_trained:
            raise TypeError("the model needs training first")
        return f(model, *args, **kwargs)
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

    __data_attributes = ()
    __trained_attributes = ()
    __forbidden_label_characters = None
    
    def __init__(self, labels, fluxes, flux_uncertainties, dispersion=None,
        threads=1, pool=None, live_dangerously=False):

        self._training_labels = labels
        self._training_fluxes = np.atleast_2d(fluxes)
        self._training_flux_uncertainties = np.atleast_2d(flux_uncertainties)
        
        self._trained = False
        self._training_set_mask = np.zeros(len(labels), dtype=bool)
        self._dispersion = np.arange(self.number_of_pixels, dtype=int) \
            if dispersion is None else dispersion
        
        # The training data must be checked, but users can live dangerously if
        # they think they can correctly specify the label vector description.
        self._verify_training_data()
        if not live_dangerously:
            self._verify_label_names()

        self.reset()

        self.threads = threads
        self.pool = pool or mp.Pool(threads) if threads > 1 else None


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

        if len(dispersion) != self.number_of_pixels:
            raise ValueError("dispersion provided does not match the number "
                             "of pixels per star ({0} != {1})".format(
                                len(dispersion), self.number_of_pixels))
        self._dispersion = dispersion


    # Attributes related to the training data.
    @property
    def training_labels(self):
        return self._training_labels


    @property
    def training_fluxes(self):
        return self._training_fluxes


    @property
    def training_flux_uncertainties(self):
        return self._training_flux_uncertainties


    @property
    def number_of_pixels(self):
        """
        Return the number of pixels for each star.
        """
        return self.training_fluxes.shape[1]


    @property
    def training_set_size(self):
        """
        Return the number of unmasked objects in the training set.
        """
        return self.get_training_set_size()


    def get_training_set_size(self, include_masked=False):
        return sum(~self.training_set_mask) if include_masked \
                                            else self.training_fluxes.shape[0]

    @property
    def training_set_mask(self):
        return self._training_set_mask


    @training_set_mask.setter
    def training_set_mask(self, mask):
        """
        Set a mask for the training set.
        """
        try:
            len(mask)
        except TypeError:
            raise TypeError("mask must be an array of the same length as the "
                            "stars in the training set")

        if mask.size != len(labels):
            raise ValueError("mask must be an array of the same length as the "
                             "stars in the training set")
        self._training_set_mask = mask


    @property
    @requires_training_wheels
    def training_data_hash(self):
        """
        A concatenated string of 10-length hashes for each item in the training
        data set.
        """
        return utils.short_hash(
            getattr(self, attr) for attr in self.__data_attributes)


    @property
    def is_trained(self):
        return self._trained


    # Attributes related to the labels and the label vector description.
    @property
    def label_names(self):
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
        for parameter in self.get_parameters(label_vector):
            if parameter not in self.label_names:
                raise ValueError("param '{0}' in the label vector description "
                                 "is not present in the training set of labels")
        self._label_vector = label_vector


    @property
    def human_readable_label_vector(self):
        """ Return a human-readable form of the label vector. """
        return utils.human_readable_label_vector(self.label_vector)


    @property
    def parameters(self):
        """ The parameters that contribute to the label vector. """
        return self.get_parameters(self.label_vector)
    

    def get_parameters(self, label_vector):
        """
        Return the parameters that contribute to the structured label vector
        provided.
        """
        return () if label_vector is None else \
            list(OrderedDict.fromkeys([label for term in label_vector \
                for label, power in term if power != 0]))

    @property
    def number_of_parameters(self):
        """ The number of parameters in the model. """
        return len(self.parameters)


    @property
    def parameter_vector(self):
        """
        Return an array of parameters of all unmasked stars in the training set.
        """
        X = np.zeros((
            self.get_training_set_size(include_masked=True),
            self.number_of_parameters))

        for i, parameter in enumerate(self.parameters):
            X[:, i] = np.array(self.training_labels[parameter])

        return X[~self.training_set_mask, :]


    # Trained attributes that subclasses are likely to use.
    # TODO: these require setters that check the entries of the data.
    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        if coefficients is None:
            self._coefficients = None
            return

        raise NotImplementedError


    @property
    def scatter(self):
        return self._scatter

    @property
    def pivot_offsets(self):
        return self._pivot_offsets


    @property
    def training_label_residuals(self):
        """
        Label residuals for stars in the training set.
        """
        if not hasattr(self, "_training_label_residuals"):
            self._training_label_residuals = self.get_training_label_residuals()
        return self._training_label_residuals


    @requires_training_wheels
    def get_training_label_residuals(self):
        """
        Return the residuals (model - true) between the parameters that the
        model returns for each star, and the believed value.
        """
        inferred_parameter_vector = np.zeros((
            self.get_training_set_size(include_masked=True),
            self.number_of_parameters))

        for i, data in \
        enumerate(zip(self.training_fluxes, self.training_flux_uncertainties)):
            if not self.training_set_mask[i]:
                inferred_parameter_vector[i, :] = self.solve_labels(*data)

        label_residuals = inferred_parameter_vector[self.training_set_mask] \
            - self.parameter_vector
        return label_residuals


    def reset(self):
        # Clear any attributes that were trained upon.

        self._trained = False
        for attr in self.__trained_attributes:
            setattr(self, "_{0}".format(attr), None)

        try:
            del self._training_label_residuals
        except AttributeError:
            None

        return None


    def _verify_label_names(self):
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
            if dispersion.size != self.number_of_pixels:
                raise ValueError(
                    "mis-match between the number of wavelength "
                    "points ({N_wls}) and flux values ({N_pxs})".format(
                        N_pxs=self.number_of_pixels, N_wls=dispersion.size))
        return None


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

        contents = [getattr(self, attr) for attr in self.__training_attributes]
        contents += [self.training_data_hash]

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


    # Methods which must be implemented by the subclasses.
    def train(self, *args, **kwargs):
        raise NotImplementedError("The train method must be "
                                  "implemented by subclasses")


    def predict(self, *args, **kwargs):
        raise NotImplementedError("The predict method must be "
                                  "implemented by subclasses")


    def solve_labels(self, *args, **kwargs):
        raise NotImplementedError("The solve_labels method must be "
                                  "implemented by subclasses")
