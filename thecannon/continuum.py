#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Continuum-normalization.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["normalize", "sines_and_cosines"]

import logging
import numpy as np


def _continuum_design_matrix(dispersion, L, order):
    """
    Build a design matrix for the continuum determination, using sines and
    cosines.

    :param dispersion:
        An array of dispersion points.

    :param L:
        The length-scale for the sine and cosine functions.

    :param order:
        The number of sines and cosines to use in the fit.
    """

    L, dispersion = float(L), np.array(dispersion)
    scale = 2 * (np.pi / L)
    return np.vstack([
        np.ones_like(dispersion).reshape((1, -1)), 
        np.array([
            [np.cos(o * scale * dispersion), np.sin(o * scale * dispersion)] \
            for o in range(1, order + 1)]).reshape((2 * order, dispersion.size))
        ])


def sines_and_cosines(dispersion, flux, ivar, continuum_pixels, L=1400, order=3, 
    regions=None, fill_value=1.0, **kwargs):
    """
    Fit the flux values of pre-defined continuum pixels using a sum of sine and
    cosine functions.

    :param dispersion:
        The dispersion values.

    :param flux:
        The flux values for all pixels, as they correspond to the `dispersion`
        array.

    :param ivar:
        The inverse variances for all pixels, as they correspond to the
        `dispersion` array.

    :param continuum_pixels:
        A mask that selects pixels that should be considered as 'continuum'.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains 
        metadata about the fit.
    """

    scalar = kwargs.pop("__magic_scalar", 1e-6) # MAGIC
    flux, ivar = np.atleast_2d(flux), np.atleast_2d(ivar)

    if regions is None:
        regions = [(dispersion[0], dispersion[-1])]

    region_masks = []
    region_matrices = []
    continuum_masks = []
    continuum_matrices = []
    pixel_included_in_regions = np.zeros_like(flux).astype(int)
    for start, end in regions:

        # Build the masks for this region.
        si, ei = np.searchsorted(dispersion, (start, end))
        region_mask = (end >= dispersion) * (dispersion >= start)
        region_masks.append(region_mask)
        pixel_included_in_regions[:, region_mask] += 1

        continuum_masks.append(continuum_pixels[
            (ei >= continuum_pixels) * (continuum_pixels >= si)])

        # Build the design matrices for this region.
        region_matrices.append(
            _continuum_design_matrix(dispersion[region_masks[-1]], L, order))
        continuum_matrices.append(
            _continuum_design_matrix(dispersion[continuum_masks[-1]], L, order))

        # TODO: ISSUE: Check for overlapping regions and raise an warning.

    # Check for non-zero pixels (e.g. ivar > 0) that are not included in a
    # region. We should warn about this very loudly!
    warn_on_pixels = (pixel_included_in_regions == 0) * (ivar > 0)

    metadata = []
    continuum = np.ones_like(flux) * fill_value
    for i in range(flux.shape[0]):

        warn_indices = np.where(warn_on_pixels[i])[0]
        if any(warn_indices):
            # Split by deltas so that we give useful warning messages.
            segment_indices = np.where(np.diff(warn_indices) > 1)[0]
            segment_indices = np.sort(np.hstack(
                [0, segment_indices, segment_indices + 1, len(warn_indices)]))
            segment_indices = segment_indices.reshape(-1, 2)

            segments = ", ".join(["{:.1f} to {:.1f} ({:d} pixels)".format(
                dispersion[s], dispersion[e], e-s) for s, e in segment_indices])

            logging.warn("Some pixels in spectrum index {0} have measured flux "
                         "values (e.g., ivar > 0) but are not included in any "
                         "specified continuum region. These pixels won't be "
                         "continuum-normalised: {1}".format(i, segments))
            
        # Get the flux and inverse variance for this object.
        object_metadata = []
        object_flux, object_ivar = (flux[i], ivar[i])

        # Normalize each region.
        for region_mask, region_matrix, continuum_mask, continuum_matrix in \
        zip(region_masks, region_matrices, continuum_masks, continuum_matrices):
            if continuum_mask.size == 0:
                # Skipping..
                object_metadata.append([order, L, fill_value, scalar, [], None])
                continue

            # We will fit to continuum pixels only.   
            continuum_disp = dispersion[continuum_mask] 
            continuum_flux, continuum_ivar \
                = (object_flux[continuum_mask], object_ivar[continuum_mask])

            # Solve for the amplitudes.
            M = continuum_matrix
            MTM = np.dot(M, continuum_ivar[:, None] * M.T)
            MTy = np.dot(M, (continuum_ivar * continuum_flux).T)

            eigenvalues = np.linalg.eigvalsh(MTM)
            MTM[np.diag_indices(len(MTM))] += scalar * np.max(eigenvalues)
            eigenvalues = np.linalg.eigvalsh(MTM)
            condition_number = max(eigenvalues)/min(eigenvalues)

            amplitudes = np.linalg.solve(MTM, MTy)
            continuum[i, region_mask] = np.dot(region_matrix.T, amplitudes)
            object_metadata.append(
                (order, L, fill_value, scalar, amplitudes, condition_number))

        metadata.append(object_metadata)

    return (continuum, metadata) 
    

def normalize(dispersion, flux, ivar, continuum_pixels, L=1400, order=3, 
    regions=None, fill_value=1.0, **kwargs):
    """
    Pseudo-continuum-normalize the flux using a defined set of continuum pixels
    and a sum of sine and cosine functions.

    :param dispersion:
        The dispersion values.

    :param flux:
        The flux values for all pixels, as they correspond to the `dispersion`
        array.

    :param ivar:
        The inverse variances for all pixels, as they correspond to the
        `dispersion` array.

    :param continuum_pixels:
        A mask that selects pixels that should be considered as 'continuum'.

    :param L: [optional]
        The length scale for the sines and cosines.

    :param order: [optional]
        The number of sine/cosine functions to use in the fit.

    :param regions: [optional]
        Specify sections of the spectra that should be fitted separately in each
        star. This may be due to gaps between CCDs, or some other physically-
        motivated reason. These values should be specified in the same units as
        the `dispersion`, and should be given as a list of `[(start, end), ...]`
        values. For example, APOGEE spectra have gaps near the following
        wavelengths which could be used as `regions`:

        >> regions = ([15090, 15822], [15823, 16451], [16452, 16971])

    :param fill_value: [optional]
        The continuum value to use for when no continuum was calculated for that
        particular pixel (e.g., the pixel is outside of the `regions`).

    :param full_output: [optional]
        If set as True, then a metadata dictionary will also be returned.

    :returns:
        The continuum values for all pixels, and a dictionary that contains 
        metadata about the fit.
    """
    continuum, metadata = sines_and_cosines(dispersion, flux, ivar, 
        continuum_pixels, L=L, order=order, regions=regions,
        fill_value=fill_value, **kwargs)

    normalized_flux = flux/continuum
    normalized_ivar = continuum * ivar * continuum
    normalized_flux[normalized_ivar == 0] = 1.0
    
    non_finite_pixels = ~np.isfinite(normalized_flux)
    normalized_flux[non_finite_pixels] = 1.0
    normalized_ivar[non_finite_pixels] = 0.0

    return (normalized_flux, normalized_ivar, continuum, metadata)


