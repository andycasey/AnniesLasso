#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the base model class and associated functions.
"""

import numpy as np
import unittest
from AnniesLasso import model, utils


class NullObject(object):
    pass


class TestRequiresTrainingWheels(unittest.TestCase):
    def test_not_trained(self):
        o = NullObject()
        o.is_trained = False
        with self.assertRaises(TypeError):
            model.requires_training_wheels(lambda x: None)(o)

    def test_is_trained(self):
        o = NullObject()
        o.is_trained = True
        self.assertIsNone(model.requires_training_wheels(lambda x: None)(o))


class TestRequiresLabelVector(unittest.TestCase):
    def test_with_label_vector(self):
        o = NullObject()
        o._descriptive_attributes = ["label_vector"]
        o.label_vector = ""
        self.assertIsNone(model.requires_model_description(lambda x: None)(o))

    def test_without_label_vector(self):
        o = NullObject()
        o._descriptive_attributes = ["label_vector"]
        o.label_vector = None
        with self.assertRaises(TypeError):
            model.requires_model_description(lambda x: None)(o)


class TestBaseCannonModel(unittest.TestCase):

    def setUp(self):
        # Initialise some faux data and labels.
        labels = "ABCDE"
        N_labels = len(labels)
        N_stars = np.random.randint(10, 500)
        N_pixels = np.random.randint(1, 10000)
        shape = (N_stars, N_pixels)

        self.valid_training_labels = np.rec.array(
            np.random.uniform(low=0.5, high=1.5, size=(N_stars, N_labels)),
            dtype=[(label, '<f8') for label in labels])

        self.valid_fluxes = np.random.uniform(low=0.5, high=1.5, size=shape)
        self.valid_flux_uncertainties = np.random.uniform(low=0.5, high=1.5,
            size=shape)

    def runTest(self):
        None
    
    def get_model(self, **kwargs):
        return model.BaseCannonModel(
            self.valid_training_labels, self.valid_fluxes,
            self.valid_flux_uncertainties, **kwargs)

    def test_repr(self):
        self.runTest() # Just for that 100%, baby.
        m = self.get_model()
        print("{0} {1}".format(m.__str__(), m.__repr__()))

    def test_get_dispersion(self):
        m = self.get_model()
        self.assertSequenceEqual(
            tuple(m.dispersion), 
            tuple(np.arange(self.valid_fluxes.shape[1])))

    def test_set_dispersion(self):
        m = self.get_model()
        for item in (None, False, True):
            # Incorrect data type (not an iterable)
            with self.assertRaises(TypeError):
                m.dispersion = item

        for item in ("", {}, [], (), set()):
            # These are iterable but have the wrong lengths.
            with self.assertRaises(ValueError):
                m.dispersion = item

        with self.assertRaises(ValueError):
            m.dispersion = [3,4,2,1]

        # These should work.
        m.dispersion = 10 + np.arange(self.valid_fluxes.shape[1])
        m.dispersion = -100 + np.arange(self.valid_fluxes.shape[1])
        m.dispersion = 520938.4 + np.arange(self.valid_fluxes.shape[1])

        # Disallow non-finite numbers.
        with self.assertRaises(ValueError):
            d = np.arange(self.valid_fluxes.shape[1], dtype=float)
            d[0] = np.nan
            m.dispersion = d

        with self.assertRaises(ValueError):
            d = np.arange(self.valid_fluxes.shape[1], dtype=float)
            d[0] = np.inf
            m.dispersion = d

        with self.assertRaises(ValueError):
            d = np.arange(self.valid_fluxes.shape[1], dtype=float)
            d[0] = -np.inf
            m.dispersion = d

        # Disallow non-float like things.
        with self.assertRaises(ValueError):
            d = np.array([""] * self.valid_fluxes.shape[1])
            m.dispersion = d

        with self.assertRaises(ValueError):
            d = np.array([None] * self.valid_fluxes.shape[1])
            m.dispersion = d
        
    def test_get_training_data(self):
        m = self.get_model()
        self.assertIsNotNone(m.training_labels)
        self.assertIsNotNone(m.training_fluxes)
        self.assertIsNotNone(m.training_flux_uncertainties)

    def test_invalid_label_names(self):
        m = self.get_model()
        for character in m._forbidden_label_characters:

            invalid_labels = [] + list(m.labels_available)
            invalid_labels[0] = "HELLO_{}".format(character)

            N_stars = len(self.valid_training_labels)
            N_labels = len(invalid_labels)
            invalid_training_labels = np.rec.array(
                np.random.uniform(size=(N_stars, N_labels)),
                dtype=[(l, '<f8') for l in invalid_labels])

            m = model.BaseCannonModel(invalid_training_labels,
                self.valid_fluxes, self.valid_flux_uncertainties,
                live_dangerously=True)

            m._forbidden_label_characters = None
            self.assertTrue(m._verify_labels_available())

            with self.assertRaises(ValueError):
                m = model.BaseCannonModel(invalid_training_labels,
                    self.valid_fluxes, self.valid_flux_uncertainties)

    def test_get_label_vector(self):
        m = self.get_model()
        m.label_vector = "A + B + C"
        self.assertEqual(m.pixel_label_vector(1), [
            [("A", 1)],
            [("B", 1)],
            [("C", 1)]
        ])

    def test_set_label_vector(self):
        m = self.get_model()
        label_vector = "A + B + C + D + E"

        m.label_vector = label_vector
        self.assertEqual(m.label_vector, utils.parse_label_vector(label_vector))
        self.assertEqual("1 + A + B + C + D + E", m.human_readable_label_vector)

        with self.assertRaises(ValueError):
            m.label_vector = "A + G"

        m.label_vector = None
        self.assertIsNone(m.label_vector)

        for item in (True, False, 0, 1.0):
            with self.assertRaises(TypeError):
                m.label_vector = item

    def test_label_getsetters(self):

        m = self.get_model()
        self.assertEqual((), m.labels)

        m.label_vector = "A + B + C"
        self.assertSequenceEqual(("A", "B", "C"), tuple(m.labels))

        with self.assertRaises(AttributeError):
            m.labels = None

    def test_inheritence(self):
        m = self.get_model()
        m.label_vector = "A + B + C"
        with self.assertRaises(NotImplementedError):
            m.train()
        m._trained = True
        with self.assertRaises(NotImplementedError):
            m.predict()
        with self.assertRaises(NotImplementedError):
            m.fit()
        
    def test_get_label_indices(self):
        m = self.get_model()
        m.label_vector = "A^5 + A^2 + B^3 + C + C*D + D^6"
        self.assertEqual([1, 2, 3, 5], m._get_lowest_order_label_indices())

    def test_data_verification(self):
        m = self.get_model()
        m._training_fluxes = m._training_fluxes.reshape(1, -1)
        with self.assertRaises(ValueError):
            m._verify_training_data()

        m._training_flux_uncertainties = \
            m._training_flux_uncertainties.reshape(m._training_fluxes.shape)
        with self.assertRaises(ValueError):
            m._verify_training_data()

        with self.assertRaises(ValueError):
            m = self.get_model(dispersion=[1,2,3])

        N_labels = 2
        N_stars, N_pixels = self.valid_fluxes.shape
        invalid_training_labels = np.random.uniform(
            low=0.5, high=1.5, size=(N_stars, N_labels))
        with self.assertRaises(ValueError):
            m = model.BaseCannonModel(invalid_training_labels,
                self.valid_fluxes, self.valid_flux_uncertainties)

    def test_labels_array(self):
        m = self.get_model()
        m.label_vector = "A^2 + B^3 + C^5"

        for i, label in enumerate("ABC"):
            foo = m.labels_array[:, i]
            bar = np.array(m.training_labels[label]).flatten()
            self.assertTrue(np.allclose(foo, bar))

        with self.assertRaises(AttributeError):
            m.labels_array = None

    def test_label_vector_array(self):
        m = self.get_model()
        m.label_vector = "A^2.0 + B^3.4 + C^5"
        m.pivots = np.zeros(len(m.labels))
        
        self.assertTrue(np.allclose(
            np.array(m.training_labels["A"]**2).flatten(),
            m.label_vector_array[1],
        ))
        self.assertTrue(np.allclose(
            np.array(m.training_labels["B"]**3.4).flatten(),
            m.label_vector_array[2]
        ))
        self.assertTrue(np.allclose(
            np.array(m.training_labels["C"]**5).flatten(),
            m.label_vector_array[3]
        ))

        m.training_labels["A"][0] = np.nan
        m.label_vector_array # For Coveralls.

        kwd1 = {
            "A": float(m.training_labels["A"][1]),
            "B": float(m.training_labels["B"][1]),
            "C": float(m.training_labels["C"][1])
        }
        kwd2 = {
            "A": [m.training_labels["A"][1][0]],
            "B": [m.training_labels["B"][1][0]],
            "C": [m.training_labels["C"][1][0]]
        }
        self.assertTrue(np.allclose(
            model._build_label_vector_rows(m.label_vector, kwd1),
            model._build_label_vector_rows(m.label_vector, kwd2)
        ))

    def test_format_input_labels(self):
        m = self.get_model()
        m.label_vector = "A^2.0 + B^3.4 + C^5"

        kwds = {"A": [5], "B": [3], "C": [0.43]}
        for k, v in m._format_input_labels(None, **kwds).items():
            self.assertEqual(kwds[k], v)
        for k, v in m._format_input_labels([5, 3, 0.43]).items():
            self.assertEqual(kwds[k], v)

        kwds_input = {k: v[0] for k, v in kwds.items() }
        for k, v in m._format_input_labels(None, **kwds_input).items():
            self.assertEqual(kwds[k], v)




    # The trained attributes and I/O functions will be tested in the sub-classes
