#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Regularized Cannon model class and associated functions.
"""

import numpy as np
import unittest
from AnniesLasso import regularized, utils


class TestRegularizedCannonModel(unittest.TestCase):

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
        return regularized.RegularizedCannonModel(
            self.valid_training_labels, self.valid_fluxes,
            self.valid_flux_uncertainties)

    def test_init(self):
        self.assertIsNotNone(self.get_model())

    def test_remind_myself_to_write_unit_tests_for_these_functions(self):
        m = self.get_model()
        m.label_vector = "A + B + C"
        self.assertIsNotNone(m.label_vector)

        # Cannot train without regularization term.
        with self.assertRaises(TypeError):
            m.train()

        # Regularization must be positive and finite.
        for each in (-1, np.nan, +np.inf, -np.inf):
            with self.assertRaises(ValueError):
                m.regularization = each

        # Regularization must be a float or match the dispersion size.
        with self.assertRaises(ValueError):
            m.regularization = [0., 1.]
        m.regularization = np.zeros_like(m.dispersion)
        m.train()
