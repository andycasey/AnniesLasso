#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functionality to continuum-normalize spectra.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.io import fits


def build_continuum_design_matrix(disp, L, order):
    """
    Build a design matrix for the continuum determination.
    """

    L = float(L)
    disp = np.array(disp)
    scale = 2 * (np.pi / L)
    return np.vstack([
        np.ones_like(disp).reshape((1, -1)), 
        np.array([[np.cos(o * scale * disp), np.sin(o * scale * disp)] \
            for o in range(1, order + 1)]).reshape((2 * order, disp.size))
        ])


def fit_continuum(dispersion, normalized_flux, normalized_ivar, continuum_pixels,
    L=1400, order=3, regions=None, fill_value=1.0, full_output=False, **kwargs):
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

    normalized_flux = np.atleast_2d(normalized_flux)
    normalized_ivar = np.atleast_2d(normalized_ivar)

    N = normalized_flux.shape[0]
    if regions is None:
        regions = [(dispersion[0], dispersion[-1])]

    """
    # Look for bad edge pixels?
    if kwargs.pop("__fix_edges", True):
        # MAGIC below.
        below_median_fraction = 0.75
        within_edge_fraction = 0.10

        for i, object_flux in enumerate(normalized_flux):
            for start, end in regions:
                si, ei = np.searchsorted(dispersion, (start, end))
                median_flux = np.median(object_flux[si:ei])

                # Any continuum points below this magic number?
                continuum_disps = dispersion[continuum_pixels]
                ok = np.where((end >= continuum_disps) \
                    * (continuum_disps >= start))[0]
                
                bad = object_flux[continuum_pixels][ok] < (median_flux * below_median_fraction)

                bad_points_at = np.arange(dispersion.size)[continuum_pixels][ok][np.where(bad)[0]]
                edge_fraction = (dispersion[bad_points_at] - start)/(end - start)
 
                bad_points_at = bad_points_at[
                    (edge_fraction < within_edge_fraction) + \
                    (edge_fraction > (1 - within_edge_fraction))]

                if np.any(bad_points_at):
                    normalized_ivar[i, bad_points_at] = 0.0
                    print("set ivar = 0 for bad points:")
                    print(dispersion[bad_points_at])
                    print(bad_points_at)
    """

    scalar = kwargs.pop("__magic_scalar", 1e-6)

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
                print("Skipping")
                object_metadata.append([order, L, fill_value, scalar, [], None])
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
            MTM[np.diag_indices(len(MTM))] += scalar * np.max(eigenvalues)#MAGIC
            eigenvalues = np.linalg.eigvalsh(MTM)
            condition_number = max(eigenvalues)/min(eigenvalues)

            amplitudes = np.linalg.solve(MTM, MTy)
            continuum[i, region_mask] = np.dot(region_matrix.T, amplitudes)
            object_metadata.append(
                (order, L, fill_value, scalar, amplitudes, condition_number))

        metadata.append(object_metadata)

    if full_output:
        return (continuum, metadata)
    return continuum



def normalize_individual_visit(dispersion, apStar_flux, apStar_ivar,
    apStar_bitmask, continuum_pixels, conservatism=(2.0, 0.1),
    normalized_ivar_floor=1e-4,
    **kwargs): # MAGIC
    """
    Stack an invividual visit from an apStar file, while properly accounting for
    the inverse variances.

    RTFD.

    Note: Revise ivar floor in 2027.
    """

    assert dispersion.size == apStar_flux.size
    assert apStar_flux.ndim == 1
    assert apStar_flux.shape == apStar_ivar.shape


    # Re-weight bad pixels based on their distance to the median.
    bad = apStar_bitmask > 0
    median_flux = np.median(apStar_flux)

    deltas = np.max(np.array([
            conservatism[0] * np.abs(apStar_flux[bad] - median_flux),
            conservatism[1] * median_flux * np.ones(bad.sum())
        ]), axis=0)
    adjusted_ivar = apStar_ivar
    adjusted_ivar[bad] = apStar_ivar[bad] / (1. + deltas**2 * apStar_ivar[bad])

    # Fit continuum first.
    kwds = kwargs.copy()
    kwds["full_output"] = False
    continuum = fit_continuum(dispersion, apStar_flux, adjusted_ivar,
        continuum_pixels, **kwds)

    # Flatten the continuum since fit_continuum can take many spectra at once.
    continuum = continuum.flatten()
    normalized_flux = apStar_flux / continuum
    # We do continuum * adj_ivar * continuum instead of continuum**2 to account
    # for the super high S/N spectra, where continuum**2 --> inf.
    normalized_ivar = continuum * adjusted_ivar * continuum

    # Clean up bad pixels.
    bad = (normalized_ivar < normalized_ivar_floor) \
        + ~np.isfinite(normalized_flux * normalized_ivar)
    normalized_flux[bad] = 1.0
    normalized_ivar[bad] = normalized_ivar_floor

    zero = normalized_flux == 0
    normalized_flux[zero] = 1.0
    normalized_ivar[zero] = 0.0

    return (normalized_flux, normalized_ivar)


def normalize_individual_visits(filename, continuum_pixels, 
    ignore_bitmask_values=(9, 10, 11), full_output=False, **kwargs):
    """
    Stack individual visits in a given apStar file.
    """

    # Extensions for the apStar files.
    ext_flux, ext_error, ext_bitmask = (1, 2, 3) # Easy as A, B, C.

    image = fits.open(filename)
    flux_array = np.atleast_2d(image[ext_flux].data)
    error_array = np.atleast_2d(image[ext_error].data)
    bitmask_array = np.atleast_2d(image[ext_bitmask].data)

    # Fix this.
    if ignore_bitmask_values is not None:
        for b in ignore_bitmask_values:
            bad = (bitmask_array & 2**b) > 0
            bitmask_array[bad] -= 2**b

    N_visits = max([1, flux_array.shape[0] - 2])
    offset = 2 if N_visits > 1 else 0

    # Calculate the dispersion array.
    dispersion = 10**(image[1].header["CRVAL1"] + \
        np.arange(flux_array.shape[1]) * image[1].header["CDELT1"])

    # Normalize the individual visit spectra.
    normalized_visit_flux = np.zeros((N_visits, dispersion.size))
    normalized_visit_ivar = np.zeros((N_visits, dispersion.size))

    metadata = {"SNR": []}
    for i in range(N_visits):

        # The first two indices contain stacked spectra with incorrect weights.
        flux = flux_array[offset + i]
        ivar = 1.0/(error_array[offset + i])**2
        bitmask = bitmask_array[offset + i]

        normed_flux, normed_ivar = normalize_individual_visit(dispersion,
            flux, ivar, bitmask, continuum_pixels, **kwargs)

        normalized_visit_flux[i, :] = normed_flux
        normalized_visit_ivar[i, :] = normed_ivar
        metadata["SNR"].append(image[0].header["SNRVIS{}".format(i+1)])

    numerator = np.sum(normalized_visit_flux * normalized_visit_ivar, axis=0)
    denominator = np.sum(normalized_visit_ivar, axis=0)

    stacked = (numerator/denominator, denominator)
    if full_output:
        return (stacked, (normalized_visit_flux, normalized_visit_ivar), metadata)
    return stacked

    """

    # Stack the individual visits.
    fig, ax = plt.subplots()
    error = 1.0/np.sqrt(normalized_visit_ivar[2])
    ax.plot(dispersion, normalized_visit_flux[2] + error, drawstyle='steps-mid')
    ax.plot(dispersion, normalized_visit_flux[2] - error, drawstyle='steps-mid')

    bad = bitmask_array[2 + offset] > 0
    ax.scatter(dispersion[bad], np.ones(bad.sum()), facecolor='r')

    fig, ax = plt.subplots()
    err = 1.0/np.sqrt(stacked_ivar)
    ax.plot(dispersion, stacked_flux, drawstyle='steps-mid', c='k')
    ax.plot(dispersion, stacked_flux - err, c="#666666", drawstyle='steps-mid',
        zorder=-1)
    ax.plot(dispersion, stacked_flux + err, c="#666666", drawstyle='steps-mid',
        zorder=-1)
    ax.set_ylim(0.7, 1.1)


    fig, ax = plt.subplots()
    for i in range(N_visits):
        ax.plot(dispersion, normalized_visit_flux[i] + i)
        bad = bitmask_array[offset + i] > 0
        ax.scatter(dispersion[bad], normalized_visit_flux[i][bad] + i, facecolor='r')






    plt.show()
    raise a
    """





def plot_continuum_fit(dispersion, flux, ivar, continuum, continuum_pixels,
    title=None):

    # Start plotting.

    fig, axes = plt.subplots(2, figsize=(14, 6))
    if title is not None:
        axes[0].set_title(title)

    # Colour continuum pixels by their relative inverse variance.
    rgba_colors = np.zeros((continuum_pixels.size, 4))
    rgba_colors[:, :3] = 0.0
    rgba_colors[:, 3] = \
        ivar[continuum_pixels]/np.median(ivar[ivar[continuum_pixels] > 0]) 
    rgba_colors = np.clip(rgba_colors, 0, 1)

    axes[0].scatter(dispersion[continuum_pixels], flux[continuum_pixels],
        c=rgba_colors, zorder=10)

    axes[0].plot(dispersion, flux, alpha=0.25, zorder=-1, c='k',
        drawstyle='steps-mid')
    
    axes[0].plot(dispersion, continuum, c='r', lw=2, zorder=1)
    axes[0].set_xticklabels([])
    axes[0].set_ylabel("Flux")

    axes[1].axhline(1, c='r', lw=2, zorder=1)
    axes[1].scatter(
        dispersion[continuum_pixels], (flux/continuum)[continuum_pixels],
        c=rgba_colors, zorder=10)

    axes[1].plot(dispersion, flux/continuum, alpha=0.25, zorder=-1, c='k',
        drawstyle='steps-mid')


    axes[0].set_xlim(dispersion[0], dispersion[-1])
    axes[1].set_xlim(dispersion[0], dispersion[-1])
    axes[0].set_ylim(0, axes[0].get_ylim()[1])
    axes[1].set_ylim(0.9, 1.1)
    axes[1].set_xlabel("Wavelength")
    axes[1].set_ylabel("Normalized flux")
    fig.tight_layout()

    return fig


if __name__ == "__main__":


    continuum_pixels = np.loadtxt("continuum.list", dtype=int)
    regions = ([15140, 15812], [15857, 16437], [16472, 16960])

    from glob import glob



    files = glob("/Users/arc/research/apogee/apogee-rg-apStar/*/*.fits")
    filename = np.random.choice(files)
    print(filename)

    filename = "/Users/arc/research/apogee/apogee-rg-apStar/4540/apStar-r5-2M19191555+1906093.fits"
    
    # This is the 8 visit stuff
    #filename = "/Users/arc/research/apogee/apogee-rg-apStar/4601/apStar-r5-2M06123730+4036001.fits"
    foo = stack_individual_visits(filename,
        continuum_pixels=continuum_pixels, regions=regions, order=2)














if __name__ == "__main__":

    """
    import os
    from sys import maxsize
    from astropy.table import Table

    # Data.
    continuum_pixels = np.loadtxt("continuum.list", dtype=int)
    PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee", "apogee-rg.fits",
        "apogee-rg-{}.memmap")

    # Load the data.
    labelled_set = Table.read(os.path.join(PATH, CATALOG))

    normalized_flux = np.memmap(
        os.path.join(PATH, FILE_FORMAT).format("normalized-flux"),
        mode="r", dtype=float).reshape((len(labelled_set), -1))
    normalized_ivar = np.memmap(
        os.path.join(PATH, FILE_FORMAT).format("normalized-ivar"),
        mode="r", dtype=float).reshape(normalized_flux.shape)
    """

    import os
    import numpy as np
    from astropy.io import fits
    from glob import glob

    PATH, FILE_FORMAT = ("/Users/arc/research/apogee", "apogee-rg-{}.memmap")

    continuum_pixels = np.loadtxt("continuum.list", dtype=int)
    dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
        mode="r", dtype=float)

    regions = ([15140, 15812], [15857, 16437], [16472, 16960])


    # Extract the combined spectra, which we will use for the training set.
    def get_combined_spectrum(filename, weighting="individual"):
        assert weighting in ("individual", "group")

        row_index = 0 if weighting == "individual" else 1
        image = fits.open(filename)
        fluxes = np.atleast_2d(image[1].data)[row_index]
        inv_var = 1.0/(np.atleast_2d(image[2].data)[row_index]**2)
        
        bad = (0 >= fluxes) + (0 >= inv_var)
        fluxes[bad] = 1.0
        inv_var[bad] = 0.0
        
        return (fluxes, inv_var, image)


    files = glob("/Users/arc/research/apogee/apogee-rg-apStar/*/*.fits")
    filename = np.random.choice(files)
    flux, ivar, image = get_combined_spectrum(filename)


    # Fit the continuum.
    continuum, flux, ivar = fit_continuum(dispersion, flux, ivar,
        continuum_pixels, regions=regions, order=4, L=1200)
    continuum, flux, ivar = [each.flatten() for each in (continuum, flux, ivar)]




    fig = plot_continuum_fit(dispersion, flux, ivar, continuum, continuum_pixels,
        title=os.path.basename(filename))
    plt.show()

    raise a


    raise a

    continuum = fit_continuum(dispersion, normalized_flux, normalized_ivar,
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

        rgba_colors = np.clip(rgba_colors, 0, 1)

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
