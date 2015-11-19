#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
General utility functions.
"""

__all__ = ["label_vector", "progressbar", "short_hash"]

import logging
import numpy as np
import sys
from time import time
from collections import Counter, OrderedDict
from itertools import combinations_with_replacement

logger = logging.getLogger(__name__)


def short_hash(contents):
    """
    Return a short hash string of some iterable content.

    :param contents:
        The contents to calculate a hash for.

    :returns:
        A concatenated string of 10-character length hashes for all items in the
        contents provided.
    """
    return "".join([str(hash(str(item)))[:10] for item in contents])


def is_structured_label_vector(label_vector):
    """
    Return whether the provided label vector is structured correctly.
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
    return True


def parse_label_vector(label_vector_description, columns=None, **kwargs):
    """
    Return a structured form of a label vector from unstructured,
    human-readable input.

    :param label_vector_description:
        A human-readable or structured form of a label vector.

    :type label_vector_description:
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

    if is_structured_label_vector(label_vector_description):
        return label_vector_description

    # Allow for custom characters, but don't advertise it.
    # (Astronomers have bad enough habits already.)
    kwds = dict(zip(("sep", "mul", "pow"), "+*^"))
    kwds.update(kwargs)
    sep, mul, pow = (kwds[k] for k in ("sep", "mul", "pow"))

    if isinstance(label_vector_description, (str, unicode)):
        label_vector_description = label_vector_description.split(sep)
    label_vector_description = map(str.strip, label_vector_description)

    # Functions to parse the parameter (or index) and order for each term.
    get_power = lambda t: int(t.split(pow)[1].strip()) if pow in t else 1
    if columns is None:
        get_label = lambda d: d.split(pow)[0].strip()
    else:
        get_label = lambda d: list(columns).index(d.split(pow)[0].strip())

    label_vector = []
    for descriptor in (item.split(mul) for item in label_vector_description):

        labels = map(get_label, descriptor)
        orders = map(get_power, descriptor)

        term = OrderedDict()
        for label, order in zip(labels, orders):
            term[label] = term.get(label, 0) + order # Sum repeat term powers.

        # Prevent uses of x^0 etc clogging up the label vector.
        label_vector.append([(l, o) for l, o in term.items() if o != 0])
    
    return label_vector


def human_readable_label_vector(label_vector, **kwargs):
    """
    Return a human-readable form of the label vector provided.
    """

    theta = ["1"]
    if label_vector is None: return theta[0]
    
    for descriptor in label_vector:
        cross_terms = []
        for label, order in descriptor:
            if order == 0: continue
            cross_terms.append(
                "".join([str(label), "^{}".format(order) if order > 1 else ""]))
        
        term = " * ".join(cross_terms)
        format = "({0})" if len(cross_terms) > 1 else "{0}"
        theta.append(format.format(term))

    return " + ".join(theta)
        

def progressbar(iterable, message=None, size=100):
    """
    A progressbar.

    :param iterable:
        Some iterable to show progress for.

    :param message: [optional]
        A string message to show as the progressbar header.

    :param size: [optional]
        The size of the progressbar. If the size given is zero or negative,
        then no progressbar will be shown.
    """

    # Preparerise.
    t_init = time()
    count = len(iterable)
    def _update(i, t=None):
        if 0 >= size: return
        increment = max(1, int(count / 100))
        if i % increment == 0 or i in (0, count):
            sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%{t}".format(
                done="=" * int(i/increment),
                not_done=" " * int((count - i)/increment),
                percent=100. * i/count,
                t="" if t is None else " ({0:.0f}s)".format(t-t_init)))
            sys.stdout.flush()

    # Initialise.
    if size > 0:
        logger.info((message or "").rstrip())
        sys.stdout.flush()

    # Updaterise.
    for i, item in enumerate(iterable):
        yield item
        _update(i)

    # Finalise.
    if size > 0:
        _update(count, time())
        sys.stdout.write("\r\n")
        sys.stdout.flush()


def label_vector(labels, order, cross_term_order=0, mul="*", pow="^"):
    """
    Build a label vector description.

    :param labels:
        The labels to use in describing the label vector.

    :param order:
        The maximum order of the terms (e.g., order 3 implies A^3 is a term).

    :param cross_term_order: [optional]
        The maximum order of the cross-terms (e.g., cross_term_order 2 implies
        A^2*B is a term).

    :param mul: [optional]
        The operator to use to represent multiplication in the description of 
        the label vector.

    :param pow: [optional]
        The operator to use to represent exponents in the description of the
        label vector.

    :returns:
        A human-readable form of the label vector.
    """

    #I make no apologies: it's fun to code like this for short complex functions
    items = []
    for o in range(1, 1 + max(order, 1 + cross_term_order)):
        for t in map(Counter, combinations_with_replacement(labels, o)):
            if len(t) == 1 and order >= max(t.values()) \
            or len(t) > 1 and cross_term_order >= max(t.values()):
                c = [pow.join([[l], [l, str(p)]][p > 1]) for l, p in t.items()]
                if c: items.append(mul.join(map(str, c)))
    return " ".join(items)
