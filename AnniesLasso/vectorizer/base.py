#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A base vectorizer for use in The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseVectorizer"]

import numpy as np


class BaseVectorizer(object):
    """
    A vectorizer class that models spectral fluxes, allows for offsets and
    scaling of different labels, and computes the derivatives for the underlying
    spectral model.

    :param label_names:
        The names of the labels that will be used by the vectorizer.

    :param fiducials:
        The fiducial offsets for the `label_names` provided.

    :param scales:
        The scaling values for the `label_names` provided.

    :param terms:
        The terms that constitute the label vector.
    """

    def __init__(self, label_names, fiducials, scales, terms):

        N = len(label_names)        
        fiducials = np.array(fiducials)
        scales = np.array(scales)

        if N != fiducials.size:
            raise ValueError("the number of fiducials does not match "
                             "the number of label_names ({0} != {1})".format(
                                N, fiducials.size))

        if N != scales.size:
            raise ValueError("the number of fiducials does not match "
                             "the number of scales {0} != {1}".format(
                                N, scales.size))

        # Fiducials can be any finite value, but scales must be finite values
        # and be positive.
        if not all(np.isfinite(fiducials)):
            raise ValueError("fiducials must be finite values")

        if not all(np.isfinite(scales)) or not all(scales > 0):
            raise ValueError("scales must be finite and positive values")

        self._label_names = label_names
        self._fiducials = fiducials
        self._scales = scales
        self._terms = terms
        return None


    # These can be over-written by sub-classes, but it is useful to have some
    # basic information if the sub-classes do not overwrite it.
    def __str__(self):
        return "<{module}.{name} object consisting of {K} labels and {D} terms>"\
            .format(module=self.__module__, name=type(self).__name__,
                K=len(self.label_names), D=len(self.terms))

    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(
            self.__module__, type(self).__name__, hex(id(self)))


    # I/O (Serializable) functionality.
    def __getstate__(self):
        """
        Return the state of the vectorizer.
        """
        return (self._label_names, self._fiducials, self._scales, self._terms)


    def __setstate__(self, state):
        """
        Set the state of the vectorizer.
        """
        self._label_names, self._fiducials, self._scales, self._terms = state


    # Read-only attributes. Don't try and change the state; create a new object.
    @property
    def label_names(self):
        """
        Return the names of the labels that contribute to the label vector.
        """
        return self._label_names


    @property
    def scales(self):
        """
        Return the scales for all labels that contribute to the label vector.
        """
        return self._scales


    @property
    def fiducials(self):
        """
        Return the fiducials for all labels that contribute to the label vector.
        """
        return self._fiducials


    @property
    def terms(self):
        """
        Return the terms provided for this vectorizer.
        """
        return self._terms


    def __call__(self, *args, **kwargs):
        """
        An alias to the get_label_vector method.
        """
        return self.get_label_vector(*args, **kwargs)


    def get_label_vector(self, labels, *args, **kwargs):
        """
        Return the label vector based on the labels provided.

        :param labels:
            The values of the labels. These should match the length and order of
            the `label_names` attribute.
        """
        raise NotImplementedError("the get_label_vector method "
                                  "must be specified by the sub-classes")


    def get_label_vector_derivative(self, labels, *args, **kwargs):
        """
        Return the derivative of the label vector with respect to the given
        label.

        :param labels:
            The values of the labels to calculate the label vector for.

        :param d_label:
            The name of the label to calculate the label vector with respect to.
        """
        raise NotImplementedError("the get_label_vector_derivative method "
                                  "must be specified by the sub-classes")


    def get_approximate_labels(self, label_vector, *args, **kwargs):
        """
        Return the approximate labels that would produce the given label_vector.
        If all terms are linearly specified in the label vector, then this is
        trivial. Otherwise, this is a per-vectorizer method.

        :param label_vector:
            The values of the label vector, typically estimated from a matrix
            inversion using observed fluxes and uncertainties.
        """
        raise NotImplementedError("the get_approximate_labels method "
                                  "must be specified by the sub-classes")


    def get_human_readable_label_vector(self, label_names=None, *args, **kwargs):
        """
        Return a human-readable form of the label vector.

        :param label_names: [optional]
            Give new label names to form the human readable label vector (e.g.,
            LaTeX label names).
        """
        raise NotImplementedError("the get_human_readable_label_vector method "
                                  "must be specified by the sub-classes")        

