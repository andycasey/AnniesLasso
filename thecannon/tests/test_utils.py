#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for general utility functions.
"""

import unittest
from .. import utils

class TestShortHash(unittest.TestCase):

    def test_different(self):
        self.assertNotEqual(utils.short_hash(True), utils.short_hash(False))

    def test_hashing(self):
        # So that it does not change with time and break old hashes.
        self.assertEqual("7fc56270e79d5ed678fe0d61f8370cf623e75af33a3ea00cfc",
            utils.short_hash("ABCDE"))