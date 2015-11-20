#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for general utility functions.
"""

import unittest
from AnniesLasso import utils


class TestShortHash(unittest.TestCase):

    def test_different(self):
        self.assertNotEqual(utils.short_hash(True), utils.short_hash(False))

    def test_hashing(self):
        # So that it does not change with time and break old hashes.
        self.assertEqual("83200250248448025411857602579487040261818832026564",
            utils.short_hash("ABCDE"))


class TestIsStructuredLabelVector(unittest.TestCase):

    def test_fail_input_string(self):
        self.assertFalse(utils.is_structured_label_vector(""))

    def test_fail_input_none(self):
        self.assertFalse(utils.is_structured_label_vector(None))

    def test_fail_input_bool(self):
        self.assertFalse(utils.is_structured_label_vector(True))
        self.assertFalse(utils.is_structured_label_vector(False))

    def test_fail_input_int(self):
        self.assertFalse(utils.is_structured_label_vector(1))

    def test_fail_input_float(self):
        self.assertFalse(utils.is_structured_label_vector(0.0))
        
    def test_fail_input_empty_iterables(self):
        self.assertFalse(utils.is_structured_label_vector(()))
        self.assertFalse(utils.is_structured_label_vector(set()))
        self.assertFalse(utils.is_structured_label_vector([]))
        self.assertFalse(utils.is_structured_label_vector({}))

    def test_fail_input_nested_empty_iterables(self):
        self.assertFalse(utils.is_structured_label_vector([{}]))
        self.assertFalse(utils.is_structured_label_vector([()]))
        self.assertFalse(utils.is_structured_label_vector([[()]]))

    def test_failed_input_term_lengths(self):
        self.assertFalse(utils.is_structured_label_vector([[("A", 0, 1)]]))
        self.assertFalse(utils.is_structured_label_vector(
            [[("A", 1)], [("B", 0, 1)]]))
    
    def test_valid_input_term_int_power(self):
        self.assertTrue(utils.is_structured_label_vector([[("A", 1)]]))

    def test_valid_input_term_float_power(self):
        self.assertTrue(utils.is_structured_label_vector([[("A", 1.0)]]))

    def test_valid_input_term_form(self):
        self.assertFalse(utils.is_structured_label_vector([[(1, "A")]]))



class TestLabelVectorParser(unittest.TestCase):

    def test_single_term(self):
        self.assertEquals([[("A", 1)]], utils.parse_label_vector("A"))

    def test_single_term_explicit(self):
        self.assertEquals([[("A", 1)]], utils.parse_label_vector("A^1"))

    def test_single_term_explicit_float(self):
        self.assertEquals([[("A", 1.2)]], utils.parse_label_vector("A^1.2"))

    def test_single_term_explicit_float_negative(self):
        self.assertEquals([[("A", -1.5)]], utils.parse_label_vector("A^-1.5"))

    def test_no_genuine_term(self):
        with self.assertRaises(ValueError):
            utils.parse_label_vector("A^0")

    def test_remove_irrelevant_terms(self):
        self.assertEquals([[("A", 2)]], utils.parse_label_vector("A^2 + B^0"))

    def test_sum_terms(self):
        self.assertEquals([[("A", 5)]], utils.parse_label_vector("A^3 * A^2"))

    def test_sum_negative_terms(self):
        self.assertEquals([[("A", -7)]], utils.parse_label_vector("A^-3 * A^-4"))
    
    def test_ignore_irrelevant_terms(self):        
        self.assertEquals([[("A", 3)]], utils.parse_label_vector("A^3 * B^0"))

    def test_parse_mixed_cross_terms(self):
        self.assertEquals([[("A", 2.0), ("B", -4.23), ("C", 1)]],
            utils.parse_label_vector("A^2.0 * B^-4.23 * C"))

    def test_invalid_powers(self):
        with self.assertRaises(ValueError):
            utils.parse_label_vector("A^B")

    def test_infinte_powers(self): # Ha!
        with self.assertRaises(ValueError):
            utils.parse_label_vector("A^inf")

        with self.assertRaises(ValueError):
            utils.parse_label_vector("A^-inf")

    def test_nan_powers(self):
        with self.assertRaises(ValueError):
            utils.parse_label_vector("A^nan")

    def test_complex_parse(self):
        self.assertEquals(utils.parse_label_vector(
            "A^4 + B^4*C^3 + D^2 *E * F^1 + G^4.3*H + J^0*G^6 * G^-6"),
            [
                [("A", 4)],
                [("B", 4), ("C", 3)],
                [("D", 2), ("E", 1), ("F", 1)],
                [("G", 4.3), ("H", 1)]
            ]
        )


class TestHumanReadableLabelVector(unittest.TestCase):

    def test_none_supplied(self):
        with self.assertRaises(TypeError):
            utils.human_readable_label_vector(None)

    def test_bool_supplied(self):
        for each in (True, False):
            with self.assertRaises(TypeError):
                utils.human_readable_label_vector(each)

    def test_str_supplied(self):
        with self.assertRaises(TypeError):
            utils.human_readable_label_vector("")

    def test_complex_parse(self):
        self.assertEqual(
            utils.human_readable_label_vector([
                [("A", 4)],
                [("B", 4), ("C", 3)],
                [("D", 2), ("E", 1), ("F", 1)],
                [("G", 4.3), ("H", 1)]
            ]),
            "1 + A^4 + (B^4 * C^3) + (D^2 * E * F) + (G^4.3 * H)"
        )


class TestProgressBar(unittest.TestCase):
    def test_iterable(self):
        for i in utils.progressbar(range(1000), size=100):
            None

    def test_nosize(self):
        for i in utils.progressbar(range(100)):
            None

    def test_message(self):
        for i in utils.progressbar(range(1000), size=100, message="Hi"):
            None

    def test_neg_size(self):
        for i in utils.progressbar(range(1000), size=-1):
            None


class TestBuildLabelVector(unittest.TestCase):
    def test_invalid_labels(self):
        self.assertEquals("", utils.build_label_vector("", 8, 1))

    def test_single_labels(self):
        self.assertEquals(
            "A + B + C + D + E",
            utils.build_label_vector("ABCDE", 1))
    
    def test_high_order_labels(self):
        self.assertEquals(
            "A + B + C + A^2 + B^2 + C^2 + A^3 + B^3 + C^3",
            utils.build_label_vector("ABC", 3))
    
    def test_single_cross_term(self):
        self.assertEquals(
            "A + B + A*B",
            utils.build_label_vector("AB", 1, 1))

    def test_high_order_cross_term(self):
        self.assertEquals(
            utils.build_label_vector("ABC", 3, 3),
            "A + B + C + A^2 + A*B + A*C + B^2 + C*B + C^2 + A^3 + A^2*B + "
            "A^2*C + A*B^2 + A*C*B + A*C^2 + B^3 + C*B^2 + C^2*B + C^3 + "
            "A^3*B + A^3*C + A^2*B^2 + A^2*C*B + A^2*C^2 + A*B^3 + A*C*B^2 "
            "+ A*C^2*B + A*C^3 + C*B^3 + C^2*B^2 + C^3*B")

    def test_different_order_cross_terms(self):
        self.assertEquals(
            utils.build_label_vector("ABC", 3, 2),
            "A + B + C + A^2 + A*B + A*C + B^2 + C*B + C^2 + A^3 + A^2*B + "
            "A^2*C + A*B^2 + A*C*B + A*C^2 + B^3 + C*B^2 + C^2*B + C^3")

