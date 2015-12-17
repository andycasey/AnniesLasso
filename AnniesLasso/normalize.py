#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A continuum normalizer.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op

def build_continuum_design_matrix(disp, L, order):
    L = float(L)
    disp = np.array(disp)
    scale = 2 * np.pi / L
    return np.vstack([
        np.ones_like(disp).reshape((1, -1)), 
        np.array([[np.cos(o * scale * disp), np.sin(o * scale * disp)] \
            for o in range(1, order + 1)]).reshape((2 * order, disp.size))
        ])

def fit(dispersion, normalized_flux, normalized_ivar, continuum_pixels, L=1400,
    order=5, regions=None, fill_value=1.0, full_output=False, **kwargs):
    """
    Fit the flux values of pre-defined continuum pixels using a sum of sine and
    cosine functions.

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
    """

    scalar = kwargs.pop("__magic_scalar", 1e-6)

    N = normalized_flux.shape[0]
    if regions is None:
        regions = [(dispersion[0], dispersion[-1])]

    continuum_masks = []
    continuum_matrices = []

    region_masks = []
    region_matrices = []

    for start, end in regions:
        # Build the masks for this region.
        si, ei = np.searchsorted(dispersion, (start, end))
        region_masks.append(
            (end >= dispersion) * (dispersion >= start))
        continuum_masks.append(continuum_pixels[
            (ei >= continuum_pixels) * (continuum_pixels >= si)])

        # Build the design matrices for this region.
        region_matrices.append(
            build_continuum_design_matrix(dispersion[region_masks[-1]], L, order))
        continuum_matrices.append(
            build_continuum_design_matrix(dispersion[continuum_masks[-1]], L, order))

        # TODO: ISSUE: Check for overlapping regions and raise an warning.

    # Keywords for optimization (except sigma).

    metadata = []
    continuum = np.ones_like(normalized_flux) * fill_value
    for i in range(N):

        print("Normalizing star {0}/{1}".format(i, N))

        # Get the flux and inverse variance for this object.
        object_metadata = []
        object_flux, object_ivar = (normalized_flux[i], normalized_ivar[i])

        # Normalize each region.
        for region_mask, region_matrix, continuum_mask, continuum_matrix \
        in zip(region_masks, region_matrices, continuum_masks, continuum_matrices):
            if continuum_mask.size == 0:
                # Give an empty list back as metadata.
                object_metadata.append([])
                continue

            # We will fit to continuum pixels.   
            continuum_disp = dispersion[continuum_mask] 
            continuum_flux, continuum_ivar \
                = (object_flux[continuum_mask], object_ivar[continuum_mask])

            # Solve for the amplitudes.
            M = continuum_matrix
            MTM = np.dot(M, continuum_ivar[:, None] * M.T)
            MTy = np.dot(M, (continuum_ivar * continuum_flux).T)

            eigenvalues = np.linalg.eigvalsh(MTM)
            MTM[np.diag_indices(len(MTM))] += scalar * np.max(eigenvalues) # MAGIC
            eigenvalues = np.linalg.eigvalsh(MTM)
            condition_number = max(eigenvalues)/min(eigenvalues)

            amplitudes = np.linalg.solve(MTM, MTy)
            continuum[i, region_mask] *= np.dot(region_matrix.T, amplitudes)

            object_metadata.append(
                (order, L, fill_value, scalar, amplitudes, condition_number))

        metadata.append(object_metadata)

    if full_output:
        return (continuum, metadata)
    return continuum



if __name__ == "__main__":


    import os
    from sys import maxsize
    from astropy.table import Table

    # Data.
    continuum_pixels = np.loadtxt("pixtest4.txt", dtype=int)
    PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee", "apogee-rg.fits",
        "apogee-rg-{}.memmap")

    # Load the data.
    labelled_set = Table.read(os.path.join(PATH, CATALOG))
    dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
        mode="r", dtype=float)
    normalized_flux = np.memmap(
        os.path.join(PATH, FILE_FORMAT).format("normalized-flux"),
        mode="r", dtype=float).reshape((len(labelled_set), -1))
    normalized_ivar = np.memmap(
        os.path.join(PATH, FILE_FORMAT).format("normalized-ivar"),
        mode="r", dtype=float).reshape(normalized_flux.shape)

    regions = ([15090, 15822], [15823, 16451], [16452, 16971])


    continuum = fit(dispersion, normalized_flux, normalized_ivar,
        continuum_pixels, regions=regions, order=3, L=1200)


    # Find the biggest example.
    mad = np.sum(np.abs(1 - continuum), axis=1)
    indices = {
        "smallest MAD": np.argmin(mad),
        "largest MAD": np.argmax(mad),
    }

    for reason, index in indices.items():

        fig, ax = plt.subplots(figsize=(14, 6))
        object_flux = normalized_flux[index]
        object_ivar = normalized_ivar[index]

        scalar = np.median(object_ivar[object_ivar > 0])

        rgba_colors = np.zeros((object_flux.size, 4))
        # Set the first three columns as zero for black.
        rgba_colors[:, :3] = 0.0
        rgba_colors[:, 3] = object_ivar/scalar # The last column is alpha.

        ax.set_title("Object index {0} has {1}".format(index, reason))

        ax.scatter(dispersion[continuum_pixels], object_flux[continuum_pixels],
            color=rgba_colors[continuum_pixels])

        ax.plot(dispersion, object_flux, alpha=0.25, zorder=-1, c='k',
            drawstyle='steps-mid')

        ax.plot(dispersion, continuum[index], c='r', lw=2)

        ax.set_ylim(0.9, 1.1)
        ax.set_xlim(dispersion[0], dispersion[-1])

        filename = "apogee-rg-normalization/normalization-test-{}.png".format(reason.replace(" ", "_"))
        fig.savefig(filename)
        print("Created {}".format(filename))


    # Draw all of the spectra?
    for index in range(normalized_flux.shape[0]):

        fig, ax = plt.subplots(figsize=(14, 6))
        object_flux = normalized_flux[index]
        object_ivar = normalized_ivar[index]

        scalar = np.median(object_ivar[object_ivar > 0])

        rgba_colors = np.zeros((object_flux.size, 4))
        # Set the first three columns as zero for black.
        rgba_colors[:, :3] = 0.0
        rgba_colors[:, 3] = object_ivar/scalar # The last column is alpha.

        ax.set_title("Object index {0}".format(index))

        ax.scatter(dispersion[continuum_pixels], object_flux[continuum_pixels],
            color=rgba_colors[continuum_pixels])

        ax.plot(dispersion, object_flux, alpha=0.25, zorder=-1, c='k',
            drawstyle='steps-mid')

        ax.plot(dispersion, continuum[index], c='r', lw=2)

        ax.set_ylim(0.9, 1.1)
        ax.set_xlim(dispersion[0], dispersion[-1])

        filename = "apogee-rg-normalization/object-{}.png".format(index)
        fig.savefig(filename)
        print("Created {}".format(filename))

        plt.close("all")
