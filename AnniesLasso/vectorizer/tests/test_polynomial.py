#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the polynomial vectorizer.
"""

import unittest
from AnniesLasso.vectorizer import polynomial

class TestIsStructuredLabelVector(unittest.TestCase):

    def test_fail_input_string(self):
        self.assertFalse(polynomial._is_structured_label_vector(""))

    def test_fail_input_none(self):
        self.assertFalse(polynomial._is_structured_label_vector(None))

    def test_fail_input_bool(self):
        self.assertFalse(polynomial._is_structured_label_vector(True))
        self.assertFalse(polynomial._is_structured_label_vector(False))

    def test_fail_input_int(self):
        self.assertFalse(polynomial._is_structured_label_vector(1))

    def test_fail_input_float(self):
        self.assertFalse(polynomial._is_structured_label_vector(0.0))
        
    def test_fail_input_empty_iterables(self):
        self.assertFalse(polynomial._is_structured_label_vector(()))
        self.assertFalse(polynomial._is_structured_label_vector(set()))
        self.assertFalse(polynomial._is_structured_label_vector([]))
        self.assertFalse(polynomial._is_structured_label_vector({}))

    def test_fail_input_nested_empty_iterables(self):
        self.assertFalse(polynomial._is_structured_label_vector([{}]))
        self.assertFalse(polynomial._is_structured_label_vector([()]))
        self.assertFalse(polynomial._is_structured_label_vector([[()]]))

    def test_failed_input_term_lengths(self):
        self.assertFalse(
            polynomial._is_structured_label_vector([[("A", 0, 1)]]))
        self.assertFalse(polynomial._is_structured_label_vector(
            [[("A", 1)], [("B", 0, 1)]]))
    
    def test_valid_input_term_int_power(self):
        self.assertTrue(polynomial._is_structured_label_vector([[("A", 1)]]))

    def test_valid_input_term_float_power(self):
        self.assertTrue(polynomial._is_structured_label_vector([[("A", 1.0)]]))

    def test_valid_input_term_form(self):
        self.assertFalse(polynomial._is_structured_label_vector([[(1, "A")]]))



class TestLabelVectorParser(unittest.TestCase):

    def test_single_term(self):
        self.assertEquals([[("A", 1)]], polynomial.parse_label_vector_description("A"))

    def test_single_term_explicit(self):
        self.assertEquals([[("A", 1)]], polynomial.parse_label_vector_description("A^1"))

    def test_single_term_explicit_float(self):
        self.assertEquals([[("A", 1.2)]], polynomial.parse_label_vector_description("A^1.2"))

    def test_single_term_explicit_float_negative(self):
        self.assertEquals([[("A", -1.5)]], polynomial.parse_label_vector_description("A^-1.5"))

    def test_no_genuine_term(self):
        with self.assertRaises(ValueError):
            polynomial.parse_label_vector_description("A^0")

    def test_remove_irrelevant_terms(self):
        self.assertEquals(
            [[("A", 2)]],
            polynomial.parse_label_vector_description("A^2 + B^0"))

    def test_sum_terms(self):
        self.assertEquals(
            [[("A", 5)]],
            polynomial.parse_label_vector_description("A^3 * A^2"))

    def test_sum_negative_terms(self):
        self.assertEquals(
            [[("A", -7)]],
            polynomial.parse_label_vector_description("A^-3 * A^-4"))
    
    def test_ignore_irrelevant_terms(self):        
        self.assertEquals(
            [[("A", 3)]],
            polynomial.parse_label_vector_description("A^3 * B^0"))

    def test_parse_mixed_cross_terms(self):
        self.assertEquals(
            [[("A", 2.0), ("B", -4.23), ("C", 1)]],
            polynomial.parse_label_vector_description("A^2.0 * B^-4.23 * C"))

    def test_invalid_powers(self):
        with self.assertRaises(ValueError):
            polynomial.parse_label_vector_description("A^B")

    def test_infinte_powers(self): # Ha!
        with self.assertRaises(ValueError):
            polynomial.parse_label_vector_description("A^inf")

        with self.assertRaises(ValueError):
            polynomial.parse_label_vector_description("A^-inf")

    def test_nan_powers(self):
        with self.assertRaises(ValueError):
            polynomial.parse_label_vector_description("A^nan")

    def test_complex_parse(self):
        self.assertEquals(polynomial.parse_label_vector_description(
            "A^4 + B^4*C^3 + D^2 *E * F^1 + G^4.3*H + J^0*G^6 * G^-6"),
            [
                [("A", 4)],
                [("B", 4), ("C", 3)],
                [("D", 2), ("E", 1), ("F", 1)],
                [("G", 4.3), ("H", 1)]
            ]
        )

    def test_valid_label_vector(self):
        label_vector = [
            [("A", 4)],
            [("B", 4), ("C", 3)],
            [("D", 2), ("E", 1), ("F", 1)],
            [("G", 4.3), ("H", 1)]
        ]
        self.assertTrue(polynomial._is_structured_label_vector(label_vector))
        self.assertEquals(label_vector, polynomial.parse_label_vector_description(label_vector))

    def test_parsing_with_columns(self):
        self.assertEquals(polynomial.parse_label_vector_description(
            "A + B^2 + C^3 + A*B", columns=["C", "A", "B"]),
            [
                [(1, 1)],
                [(2, 2)],
                [(0, 3)],
                [(1, 1), (2, 1)]
            ])


class TestHumanReadableLabelVector(unittest.TestCase):

    def test_none_supplied(self):
        with self.assertRaises(TypeError):
            polynomial.human_readable_label_vector(None)

    def test_bool_supplied(self):
        for each in (True, False):
            with self.assertRaises(TypeError):
                polynomial.human_readable_label_vector(each)

    def test_str_supplied(self):
        with self.assertRaises(TypeError):
            polynomial.human_readable_label_vector("")

    def test_complex_parse(self):
        self.assertEqual(
            polynomial.human_readable_label_vector([
                [("A", 4)],
                [("B", 4), ("C", 3)],
                [("D", 2), ("E", 1), ("F", 1)],
                [("G", 4.3), ("H", 1)]
            ]),
            "1 + A^4 + B^4*C^3 + D^2*E*F + G^4.3*H"
        )

    def test_complex_parse_with_brackets(self):
        self.assertEqual(
            polynomial.human_readable_label_vector([
                [("A", 4)],
                [("B", 4), ("C", 3)],
                [("D", 2), ("E", 1), ("F", 1)],
                [("G", 4.3), ("H", 1)]
            ], mul=" * ", bracket=True),
            "1 + A^4 + (B^4 * C^3) + (D^2 * E * F) + (G^4.3 * H)"
        )



class TestBuildLabelVector(unittest.TestCase):
    def test_invalid_labels(self):
        self.assertEquals("", polynomial.terminator("", 8, 1))

    def test_single_labels(self):
        self.assertEquals(
            "A + B + C + D + E",
            polynomial.terminator("ABCDE", 1))
    
    def test_high_order_labels(self):
        self.assertEquals(
            "A + B + C + A^2 + B^2 + C^2 + A^3 + B^3 + C^3",
            polynomial.terminator("ABC", 3, cross_term_order=0))
    
    def test_high_order_labels_and_cross_terms(self):
        self.assertEquals(
            "A + B + C + A^2 + A*B + A*C + B^2 + B*C + C^2",
            polynomial.terminator("ABC", 2, cross_term_order=-1))

    def test_single_cross_term(self):
        self.assertEquals(
            "A + B + A*B",
            polynomial.terminator("AB", 1, 1))

    def test_high_order_cross_term(self):
        self.assertEquals(
            polynomial.terminator("ABC", 3, 3),
            "A + B + C + A^2 + A*B + A*C + B^2 + B*C + C^2 + A^3 + A^2*B + "
            "A^2*C + A*B^2 + A*B*C + A*C^2 + B^3 + B^2*C + B*C^2 + C^3 + "
            "A^3*B + A^3*C + A^2*B^2 + A^2*B*C + A^2*C^2 + A*B^3 + A*B^2*C + "
            "A*B*C^2 + A*C^3 + B^3*C + B^2*C^2 + B*C^3"
        )

    def test_different_order_cross_terms(self):
        self.assertEquals(
            polynomial.terminator("ABC", 3, 2),
            "A + B + C + A^2 + A*B + A*C + B^2 + B*C + C^2 + A^3 + A^2*B + "
            "A^2*C + A*B^2 + A*B*C + A*C^2 + B^3 + B^2*C + B*C^2 + C^3")
