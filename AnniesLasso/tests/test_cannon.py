#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Cannon model class and associated functions.
"""

import numpy as np
import unittest
from six.moves import cPickle as pickle
from os import path

from AnniesLasso import cannon

# Test the individual fitting functions first, then we can generate some
# real 'fake' data for the full Cannon model test.


# Now test the other stuff

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
        with open(path.join(here, "test_data_set.pkl"), "r") as fp:
            contents = pickle.load(fp)

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


    def runTest(self):

        # Train all.
        self.do_training()

        self.do_residuals()


