#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A polynomial vectorizer for use in The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BasePolynomialVectorizer", "NormalizedPolynomialVectorizer",
    "terminator"]

import numpy as np
from collections import (Counter, OrderedDict)
from six import string_types

from .base import BaseVectorizer


class BasePolynomialVectorizer(BaseVectorizer):
    """
    A base vectorizer class that models spectral fluxes as a linear combination
    of label terms in a polynomial fashion.

    :param label_names:
        The names of the labels that will be used by the vectorizer.

    :param fiducials:
        The fiducial offsets for the labels.

    :param scales:
        The scaling values for the labels.

    :param terms:
        The terms that constitute the label vector.
    """

    def __init__(self, *args, **kwargs):
        super(BasePolynomialVectorizer, self).__init__(*args, **kwargs)

        # Check that the terms are OK, and parse them into a useful format.
        self._terms = parse_label_vector_description(
            self._terms, columns=self.label_names)
        return None


    def get_label_vector(self, labels):
        """
        Return the values of the label vector, given the labels.

        :param labels:
            The labels to calculate the label vector for. This can be a 
            one-dimensional vector of `K` labels (using the same order and
            length provided by self.label_names), or a two-dimensional array of
            `N` by `K` values. The returning array will be of shape `(N, D)`,
            where `D` is the number of terms in the label vector description.
        """

        labels = np.atleast_2d(labels)
        if labels.ndim > 2:
            raise ValueError("labels must be a 1-d or 2-d array")

        # Offset and scale the labels before building the vector.
        scaled_labels = (labels - self.fiducials)/self.scales

        columns = [np.ones(labels.shape[0], dtype=float)]
        for term in self.terms:
            column = 1. # This works; don't use np.multiply/np.product.
            for index, order in term:
                column *= scaled_labels[:, index]**order
            columns.append(column)
        return np.vstack(columns).T


    def get_approximate_labels(self, label_vector):
        """
        Return the approximate labels that would produce the given label_vector.
        If all terms are linearly specified in the label vector, then this is
        trivial. Otherwise, this is a per-vectorizer method.

        :param label_vector:
            The values of the label vector, typically estimated from a matrix
            inversion using observed fluxes and uncertainties.
        """

        # Need to match the label vector terms back to real labels.
        # (Maybe this should use some general non-linear simultaneous solver?)

        # The term_index tells us which term in the label vector is the best to
        # estimate the label from (e.g., a linear term).

        # The label_index tells us which label this term is actually
        # referring to.

        labels = np.nan * np.ones(len(self.label_names))
        for term_index in self._lowest_order_label_indices:
            if term_index is None: continue

            label_index, order = self.terms[term_index][0]
            # The +1 index offset is because the first theta is a scaling.
            labels[label_index] = abs(label_vector[1 + term_index])**(1./order)

        # There could be some coefficients that are only used in cross-terms...
        # We could solve for them, or just take them as zeros (which will then
        # put them as the fiducials)
        labels[~np.isfinite(labels)] = 0.

        return labels * self.scales + self.fiducials


    @property
    def _lowest_order_label_indices(self):
        """
        Get the indices for the lowest power label terms in the label vector.
        """
        indices = OrderedDict()
        for i, term in enumerate(self.terms):
            if len(term) > 1: continue
            index, order = term[0]
            if order < indices.get(index, [None, np.inf])[-1]:
                indices[index] = (i, order)

        return [indices.get(i, [None])[0] for i in range(len(self.label_names))]


    def get_human_readable_label_vector(self, label_names=None, mul="*", pow="^",
        **kwargs):
        """
        Return a human-readable form of the label vector.

        :param label_names: [optional]
            Give new label names to form the human readable label vector (e.g.,
            LaTeX label names).

        :param mul: [optional]
            String to use to represent a multiplication operator. For example,
            if giving LaTeX label definitions one may want to use '\cdot' for
            the `mul` term.

        :param pow: [optional]
            String to use to represent a power operator.
        """
        labels = labels or self.label_names
        terms = ["1"]
        for term in self.terms:
            cross_term = []
            for i, o in term:
                if o > 1:
                    cross_term.append("{0}{1}{2:.0f}".format(labels[i], pow, o))
                else:
                    cross_term.append(labels[i])
            terms.append(mul.join(cross_term))
        return terms


class NormalizedPolynomialVectorizer(BasePolynomialVectorizer):
    """
    A vectorizer class that models spectral fluxes as a linear combination of
    label terms in a polynomial fashion. The fiducials and scales are determined
    automatically such that each label dimension has nearly unit variance.

    :param labelled_set:
        A table containing the label values for all of the labels that will form
        this label vector. These data are used to calculate the fiducials and
        scales.

    :param terms:
        The terms that constitute the label vector.
    """

    def __init__(self, labelled_set, terms):
        # First parse the terms so that we can get the label names.
        structured_terms = parse_label_vector_description(terms)
        label_names = get_label_names(structured_terms)
        terms = parse_label_vector_description(terms, columns=label_names)

        # Ensure the requested label names are actually in the available tables.
        for label_name in label_names:
            try:
                label_table[label_name]
            except KeyError:
                raise KeyError("missing label '{}' in the table"\
                    .format(label_name))

        # Calculate the scales and fiducials.
        scales = [np.ptp(np.percentile(label_table[_], [2.1, 97.9])) \
            for _ in label_names]
        fiducials = [np.percentile(label_table[_], 50) for _ in label_names]

        super(NormalizedPolynomialVectorizer, self).__init__(
            label_names, fiducials, scales, terms)


def _is_structured_label_vector(label_vector):
    """
    Return whether the provided label vector is structured as a polynomial
    vector description appropriately or not.

    :param label_vector:
        A structured or unstructured description of a polynomial label vector.
    """
    if not isinstance(label_vector, (list, tuple)):
        return False

    for descriptor in label_vector:
        if not isinstance(descriptor, (list, tuple)):
            return False

        for term in descriptor:
            if not isinstance(term, (list, tuple)) \
            or len(term) != 2 \
            or not isinstance(term[-1], (int, float)):
                return False

    if len(label_vector) == 0 or sum(map(len, label_vector)) == 0:
        return False
    return True


def parse_label_vector_description(description, columns=None, **kwargs):
    """
    Return a structured form of a label vector from unstructured,
    human-readable input.

    :param description:
        A human-readable or structured form of a label vector.

    :type description:
        str or list

    :param columns: [optional]
        If `columns` are provided, instead of text columns being provided as the
        output parameter, the corresponding index location in `column` will be
        given.

    :returns:
        A structured form of the label vector as a multi-level list.


    :Example:

    >>> parse_label_vector("Teff^4 + logg*Teff^3 + feh + feh^0*Teff")
    [
        [
            ("Teff", 4),
        ],
        [
            ("logg", 1),
            ("Teff", 3)
        ],
        [
            ("feh", 1),
        ],
        [
            ("feh", 0),
            ("Teff", 1)
        ]
    ]
    """

    if _is_structured_label_vector(description):
        return description

    # Allow for custom characters, but don't advertise it.
    # (Astronomers have bad enough habits already.)
    kwds = dict(zip(("sep", "mul", "pow"), "+*^"))
    kwds.update(kwargs)
    sep, mul, pow = (kwds[k] for k in ("sep", "mul", "pow"))

    if isinstance(description, string_types):
        description = description.split(sep)
    description = [_.strip() for _ in description]

    # Functions to parse the parameter (or index) and order for each term.
    get_power = lambda t: float(t.split(pow)[1].strip()) if pow in t else 1
    if columns is None:
        get_label = lambda d: d.split(pow)[0].strip()
    else:
        get_label = lambda d: list(columns).index(d.split(pow)[0].strip())

    label_vector = []
    for descriptor in (item.split(mul) for item in description):

        labels = map(get_label, descriptor)
        orders = map(get_power, descriptor)

        term = OrderedDict()
        for label, order in zip(labels, orders):
            term[label] = term.get(label, 0) + order # Sum repeat term powers.

        # Prevent uses of x^0 etc clogging up the label vector.
        valid_terms = [(l, o) for l, o in term.items() if o != 0]
        if not np.all(np.isfinite([o for l, o in valid_terms])):
            raise ValueError("non-finite power provided")

        if len(valid_terms) > 0:
            label_vector.append(valid_terms)
    
    if sum(map(len, label_vector)) == 0:
        raise ValueError("no valid terms provided")

    return label_vector


def terminator(label_names, order, cross_term_order=-1, **kwargs):
    """
    Create the terms required for a label vector description based on the label
    names provided and the order given.

    :param label_names:
        The names of the labels to use in describing the label vector.

    :param order:
        The maximum order of the terms (e.g., order 3 implies A^3 is a term).

    :param cross_term_order: [optional]
        The maximum order of the cross-terms (e.g., cross_term_order 2 implies
        A^2*B is a term). If the provided `cross_term_order` value is negative, 
        then `cross_term_order = order - 1` will be assumed.

    :param mul: [optional]
        The operator to use to represent multiplication in the description of 
        the label vector.

    :param pow: [optional]
        The operator to use to represent exponents in the description of the
        label vector.

    :returns:
        A human-readable form of the label vector.
    """
    sep, mul, pow = kwargs.pop(["sep", "mul", "pow"], "+*^")

    #I make no apologies: it's fun to code like this for short complex functions
    items = []
    if 0 > cross_term_order:
        cross_term_order = order - 1

    for o in range(1, 1 + max(order, 1 + cross_term_order)):
        for t in map(Counter, combinations_with_replacement(label_names, o)):
            # Python 2 and 3 behave differently here, so generate an ordered
            # dictionary based on sorting the keys.
            t = OrderedDict([(k, t[k]) for k in sorted(t.keys())])
            if len(t) == 1 and order >= max(t.values()) \
            or len(t) > 1 and cross_term_order >= max(t.values()):
                c = [pow.join([[l], [l, str(p)]][p > 1]) for l, p in t.items()]
                if c: items.append(mul.join(map(str, c)))
    return " {} ".format(sep).join(items)


def get_label_names(label_vector):
    """
    Return the label names that contribute to the structured label vector
    description provided.

    :param label_vector:
        A structured description of the label vector.
    """
    return list(OrderedDict.fromkeys([label for term in label_vector \
        for label, power in term if power != 0]))