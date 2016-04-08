#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the base vectorizer.
"""

import os
import tempfile
import unittest
from AnniesLasso.vectorizer import base





class TestBaseVectorizerInitialization(unittest.TestCase):

    labels = ("a", "b", "c", "d", "e")

    def test_incompatible_scale_length(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels)),
                np.random.uniform(size=len(self.labels) + 1),
                [])

    def test_incompatible_fiducial_length_1(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels) - 1),
                np.random.uniform(size=len(self.labels)),
                [])


    def test_non_finite_fiducials(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                ([0] * len(self.labels)) + [np.nan],
                np.random.uniform(size=len(self.labels)),
                [])


    def test_infinite_fiducials(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                ([0] * len(self.labels)) + [+np.inf],
                np.random.uniform(size=len(self.labels)),
                [])

        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                ([0] * len(self.labels)) + [-np.inf],
                np.random.uniform(size=len(self.labels)),
                [])


    def test_non_finite_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels)),
                ([0] * len(self.labels)) + [np.nan],
                [])


    def test_infinite_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels)),
                ([0] * len(self.labels)) + [+np.inf],
                [])

        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels)),
                ([0] * len(self.labels)) + [-np.inf],
                [])


    def test_negative_scales(self):
        with self.assertRaises(ValueError):
            base.BaseVectorizer(self.labels,
                np.random.uniform(size=len(self.labels)),
                np.random.uniform([-2, -1], size=len(self.labels)),
                [])


    def test_serialization(self):
        before = base.BaseVectorizer(self.labels,
            np.random.uniform(size=len(self.labels)),
            np.random.uniform(size=len(self.labels)),
            [(1, 0), (2, 1.0)]) # ignored, just for testing.

        _, path = tempfile.mkstemp()

        with open(path, "wb") as fp:
            pickle.dump(before, fp, -1)


        with open(path, "rb") as fp:
            after = pickle.load(fp)

        # Check integrity
        self.assertEqual(before.label_names, after.label_names)
        self.assertEqual(before.scales, after.scales)
        self.assertEqual(before.fiducials, after.fiducials)
        self.assertEqual(before.terms, after.terms)

        if os.path.exists(path):
            os.remove(path)
