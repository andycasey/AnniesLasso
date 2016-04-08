#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the base vectorizer.
"""

import numpy as np
import os
import tempfile
import unittest
from six.moves import cPickle as pickle

from AnniesLasso.vectorizer import base


class TestBaseVectorizerInitialization(unittest.TestCase):

    label_names = ("a", "b", "c", "d", "e")

    def test_incompatible_scale_length(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names)),
                np.random.uniform(size=len(self.label_names) + 1),
                [])

    def test_incompatible_fiducial_length_1(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names) - 1),
                np.random.uniform(size=len(self.label_names)),
                [])


    def test_non_finite_fiducials(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                ([0] * (len(self.label_names) - 1)) + [np.nan],
                np.random.uniform(size=len(self.label_names)),
                [])


    def test_infinite_fiducials(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                ([0] * (len(self.label_names) - 1) + [+np.inf],
                np.random.uniform(size=len(self.label_names)),
                [])

        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                ([0] * (len(self.label_names) - 1) + [-np.inf],
                np.random.uniform(size=len(self.label_names)),
                [])


    def test_non_finite_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names)),
                ([0] * len(self.label_names)) + [np.nan],
                [])


    def test_infinite_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names)),
                ([0] * len(self.label_names)) + [+np.inf],
                [])

        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names)),
                ([0] * len(self.label_names)) + [-np.inf],
                [])


    def test_negative_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.label_names,
                np.random.uniform(size=len(self.label_names)),
                np.random.uniform([-2, -1], size=len(self.label_names)),
                [])


    def test_serialization(self):
        before = base.BaseVectorizer(self.label_names,
            np.random.uniform(size=len(self.label_names)),
            np.random.uniform(size=len(self.label_names)),
            [(1, 0), (2, 1.0)]) # ignored, just for testing.

        _, path = tempfile.mkstemp()

        with open(path, "wb") as fp:
            pickle.dump(before, fp, -1)


        with open(path, "rb") as fp:
            after = pickle.load(fp)

        # Check integrity
        self.assertTrue(np.all(before.label_names == after.label_names))
        self.assertTrue(np.all(before.scales == after.scales))
        self.assertTrue(np.all(before.fiducials == after.fiducials))
        self.assertTrue(np.all(before.terms == after.terms))

        # Clean up.
        if os.path.exists(path):
            os.remove(path)


    def test_transforms(self):
        v = base.BaseVectorizer(self.label_names,
            np.random.uniform(size=len(self.label_names)),
            np.random.uniform(size=len(self.label_names)),
            [])

        for labels in np.random.uniform(size=(100, len(self.label_names))):
            self.assertTrue(
                np.allclose(labels, v._inv_transform(v._transform(labels))))


    def test_no_monkey_patching(self):

        v = base.BaseVectorizer(self.label_names,
            np.random.uniform(size=len(self.label_names)),
            np.random.uniform(size=len(self.label_names)),
            [])

        for method in (v, v.get_label_vector, v.get_label_vector_derivative,
            v.get_approximate_labels, v.get_human_readable_label_vector):

            with self.assertRaises(NotImplementedError):
                method(None)


    def test_repr(self):
        v = base.BaseVectorizer(self.label_names,
            np.random.uniform(size=len(self.label_names)),
            np.random.uniform(size=len(self.label_names)),
            [])
        _ = "{}".format(v.__str__())
        __ = "{}".format(v.__repr__())


