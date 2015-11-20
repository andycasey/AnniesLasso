#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Cannon model class and associated functions.
"""

import numpy as np
import sys
import unittest
from six.moves import cPickle as pickle
from os import path, remove
from tempfile import mkstemp

from AnniesLasso import cannon, utils


class TestCannonModel(unittest.TestCase):

    def setUp(self):
        # Initialise some faux data and labels.
        labels = "ABCDE"
        N_labels = len(labels)
        N_stars = np.random.randint(1, 500)
        N_pixels = np.random.randint(1, 10000)
        shape = (N_stars, N_pixels)

        self.valid_training_labels = np.rec.array(
            np.random.uniform(size=(N_stars, N_labels)),
            dtype=[(label, '<f8') for label in labels])

        self.valid_fluxes = np.random.uniform(size=shape)
        self.valid_flux_uncertainties = np.random.uniform(size=shape)

    def get_model(self):
        return cannon.CannonModel(
            self.valid_training_labels, self.valid_fluxes,
            self.valid_flux_uncertainties)

    def test_init(self):
        self.assertIsNotNone(self.get_model())


# The test_data_set.pkl contains:
# (training_labels, training_fluxes, training_flux_uncertainties, coefficients,
#  scatter, label_vector)
# The training labels are not named, but they are: (TEFF, LOGG, PARAM_M_H)

class TestCannonModelRealistically(unittest.TestCase):

    def setUp(self):
        # Set up a model using the test data set.
        here = path.dirname(path.realpath(__file__))
        kwds = {"encoding": "utf-8", "errors": "ignore"} \
            if sys.version_info[0] >= 3 else {}
        with open(path.join(here, "test_data_set.pkl"), "rb") as fp:
            contents = pickle.load(fp, **kwds)

        # Unpack it all 
        training_labels, training_fluxes, training_flux_uncertainties, \
            coefficients, scatter, label_vector = contents

        training_labels = np.core.records.fromarrays(training_labels,
            names="TEFF,LOGG,PARAM_M_H", formats="f8,f8,f8")

        self.test_data_set = {
            "training_labels": training_labels,
            "training_fluxes": training_fluxes,
            "training_flux_uncertainties": training_flux_uncertainties,
            "coefficients": coefficients,
            "scatter": scatter,
            "label_vector": label_vector

        }
        self.model_serial = cannon.CannonModel(training_labels, training_fluxes,
            training_flux_uncertainties)
        self.model_parallel = cannon.CannonModel(training_labels,
            training_fluxes, training_flux_uncertainties, threads=2)

        self.models = (self.model_serial, self.model_parallel)

    def do_training(self):
        for model in self.models:
            model.reset()
            model.label_vector = self.test_data_set["label_vector"]
            self.assertIsNotNone(model.train())

        # Check that the trained attributes in both model are equal.
        for _attribute in self.model_serial._trained_attributes:
            self.assertTrue(np.allclose(
                getattr(self.model_serial, _attribute),
                getattr(self.model_parallel, _attribute)
                ))

            # And nearly as we expected.
            self.assertTrue(np.allclose(
                self.test_data_set[_attribute[1:]],
                getattr(self.model_serial, _attribute),
                rtol=0.5, atol=1e-8))

    def do_residuals(self):
        serial = self.model_serial.get_training_label_residuals()
        parallel = self.model_parallel.get_training_label_residuals()
        self.assertTrue(np.allclose(serial, parallel))

    def ruin_the_trained_coefficients(self):
        self.model_serial.scatter = None
        self.assertIsNone(self.model_serial.scatter)

        with self.assertRaises(ValueError):
            self.model_parallel.scatter = [3]

        for item in (0., False, True):
            with self.assertRaises(ValueError):
                self.model_parallel.scatter = item

        with self.assertRaises(ValueError):
            self.model_parallel.scatter = \
                -1 * np.ones_like(self.model_parallel.dispersion)

        _ = np.array(self.model_parallel.scatter).copy()
        _ += 1.
        self.model_parallel.scatter = _
        self.assertTrue(np.allclose(_, self.model_parallel.scatter))


        self.model_serial.coefficients = None
        self.assertIsNone(self.model_serial.coefficients)

        with self.assertRaises(ValueError):
            self.model_parallel.coefficients = np.arange(12).reshape((3, 2, 2))

        with self.assertRaises(ValueError):
            _ = np.ones_like(self.model_parallel.coefficients)
            self.model_parallel.coefficients = _.T

        with self.assertRaises(ValueError):
            _ = np.ones_like(self.model_parallel.coefficients)
            self.model_parallel.coefficients = _[:, :-1]
        
        _ = np.array(self.model_parallel.coefficients).copy()
        _ += 0.5
        self.model_parallel.coefficients = _
        self.assertTrue(np.allclose(_, self.model_parallel.coefficients))

    def do_io(self):

        _, temp_filename = mkstemp()
        remove(temp_filename)
        self.model_serial.save(temp_filename, include_training_data=False)
        with self.assertRaises(IOError):
            self.model_serial.save(temp_filename, overwrite=False)

        names = ("_data_attributes", "_trained_attributes",
            "_descriptive_attributes")
        attrs = (
            self.model_serial._data_attributes,
            self.model_serial._trained_attributes,
            self.model_serial._descriptive_attributes
            )
        for name, item in zip(names, attrs):
            _ = [] + list(item)
            _.append("metadata")
            setattr(self.model_serial, name, _)
            with self.assertRaises(ValueError):
                self.model_serial.save(temp_filename, overwrite=True)
            setattr(self.model_serial, name, _[:-1])

        self.model_serial.save(temp_filename, include_training_data=True,
            overwrite=True)

        self.model_parallel.reset()
        self.model_parallel.load(temp_filename, verify_training_data=True)

        # Check that the trained attributes in both model are equal.
        for _attribute in self.model_serial._trained_attributes:
            self.assertTrue(np.allclose(
                getattr(self.model_serial, _attribute),
                getattr(self.model_parallel, _attribute)
                ))

            # And nearly as we expected.
            self.assertTrue(np.allclose(
                self.test_data_set[_attribute[1:]],
                getattr(self.model_serial, _attribute),
                rtol=0.5, atol=1e-8))

        # Check that the data attributes in both model are equal.
        for _attribute in self.model_serial._data_attributes:
            self.assertTrue(
                utils.short_hash(getattr(self.model_serial, _attribute)),
                utils.short_hash(getattr(self.model_parallel, _attribute))
            )

        if path.exists(temp_filename):
            remove(temp_filename)

    def do_cv(self):
        self.model_parallel.cross_validate(N=1, debug=True)

    def do_predict(self):
        _ = [self.model_serial.training_labels[label][0] \
            for label in self.model_serial.labels]
        self.assertTrue(np.allclose(
            self.model_serial.predict(_),
            self.model_serial.predict(**dict(zip(self.model_serial.labels, _)))))

    def runTest(self):

        # Train all.
        self.do_training()

        self.do_residuals()

        self.ruin_the_trained_coefficients()

        # Train again.
        self.do_training()

        # Predict stuff.
        self.do_predict()

        # Do cross-validation.
        self.do_cv()

        # Try I/O/
        self.do_io()
