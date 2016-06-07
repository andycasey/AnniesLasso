#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the censoring dictionary.
"""

import numpy as np
import unittest
from collections import namedtuple

from AnniesLasso.censoring import CensorsDict


class fake_model(object):

    def __init__(self, N=1000, label_names="abcde"):

        self.dispersion = np.arange(N)
        v = namedtuple('foo', ['label_names'])
        self.vectorizer = v(label_names=label_names)
        self.censors = CensorsDict(self)


class TestCensorsDict(unittest.TestCase):

    def test_boolean_mask(self):

        N = 1000
        f = fake_model(N, label_names=["a"])

        mask = np.round(np.random.uniform(size=N)).reshape(-1, 5)
        f.censors["a"] = mask
        
        self.assertTrue(np.all(f.censors["a"] == mask.flatten().astype(bool)))


    def test_non_finite_boolean_masks(self):

        N = 1000
        f = fake_model(N, label_names=["a"])

        mask = np.round(np.random.uniform(size=N)).reshape(-1, 5)
        
        bad = (-np.inf, +np.inf, np.nan)
        for each in bad:
            mask[100, 2] = each
            with self.assertRaises(ValueError):
                f.censors["a"] = mask


    def test_incompatible_mask(self):

        N = 1000
        f = fake_model(N, label_names=["a"])

        mask = np.round(np.random.uniform(size=N/2))
        with self.assertRaises(ValueError):
            f.censors["a"] = mask


    def test_range_mask(self):

        f = fake_model()

        censored_ranges = [
            [100, 200],
            [300, 400],
            [500, 600],
            [700, 800],
            [900, 1000]
        ]

        f.censors["a"] = censored_ranges

        # The first pixels should not be censored.
        self.assertFalse(f.censors["a"][0])

        # The mid-pixel between each range should be censored.
        for start, end in censored_ranges:
            mid = start + (end - start)/2
            index = f.dispersion.searchsorted(mid)
            self.assertTrue(f.censors["a"][index])


    def test_range_mask_with_non_finites(self):
    
        f = fake_model()

        censored_ranges = [
            [-np.inf, 200],
            [300, 400],
            [500, 600],
            [700, 800],
            [900, +np.inf]
        ]

        f.censors["a"] = censored_ranges

        # The first and last pixels should be censored
        self.assertTrue(f.censors["a"][0])
        self.assertTrue(f.censors["a"][-1])


    def test_range_with_nones(self):
    
        f = fake_model()

        censored_ranges = [
            [None, 200],
            [300, 400],
            [500, 600],
            [700, 800],
            [900, None]
        ]

        f.censors["a"] = censored_ranges

        # The first and last pixels should be censored
        self.assertTrue(f.censors["a"][0])
        self.assertTrue(f.censors["a"][-1])


        censored_ranges = [
            [100, 200],
            [300, 400],
            [None, 600],
            [700, 800],
            [900, None]
        ]

        f.censors["b"] = censored_ranges
        # All 600 pixels at the start will now be censored.
        self.assertTrue(np.all(f.censors["b"][:600]))