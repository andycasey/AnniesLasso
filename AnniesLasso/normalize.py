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
    curve_fit_kwds = {
        "p0": np.ones(2*order + 1),#np.ones(2*K + 1),
        "maxfev": 10**6, # Cannot set as np.inf for maxfev.
        "absolute_sigma": False
    }
    curve_fit_kwds.update(kwargs)
    p0_theta = curve_fit_kwds["p0"]

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
            if continuum_mask.size == 0: continue

            # We will fit to continuum pixels.   
            continuum_disp = dispersion[continuum_mask] 
            continuum_flux, continuum_ivar \
                = (object_flux[continuum_mask], object_ivar[continuum_mask])


            M = continuum_matrix
            MTM = np.dot(M, continuum_ivar[:, None] * M.T)
            MTy = np.dot(M, (continuum_ivar * continuum_flux).T)

            eigenvalues = np.linalg.eigvalsh(MTM)
            print("Condition number: {}".format(max(eigenvalues)/min(eigenvalues)))
            assert np.all(eigenvalues > 0)
            MTM[np.diag_indices(len(MTM))] += 1e-6 * np.max(eigenvalues) # MAGIC
            eigenvalues = np.linalg.eigvalsh(MTM)
            print("Condition number after: {}".format(max(eigenvalues)/min(eigenvalues)))
            #assert np.all(eigenvalues > 0)


            amps = np.linalg.solve(MTM, MTy)

            cont = np.dot(region_matrix.T, amps)


            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            scale = np.median(continuum_ivar[continuum_ivar > 0])
            for i in range(continuum_disp.size):

                ax.scatter([continuum_disp[i]], [continuum_flux[i]], facecolor="k",
                    alpha=continuum_ivar[i]/scale)

            ax.plot(dispersion[region_mask], object_flux[region_mask], alpha=0.25,
                zorder=-1, c='k', drawstyle='steps-mid')

            ax.plot(dispersion[region_mask], cont)

            ax.set_ylim(0.9, 1.1)
            ax.set_xlim(dispersion[region_mask][0], dispersion[region_mask][-1])


            raise a
            continuum[i, region_mask] *= np.dot(op_theta, region_matrix)


            # First try with curve_fit.
            continuum_model = lambda x, *theta: np.dot(theta, continuum_matrix)

            curve_fit_kwds["sigma"] = continuum_ivar


            op_theta, cov = op.curve_fit(
                continuum_model, continuum_disp, continuum_flux, **curve_fit_kwds)


            """
            # And let's try Powell.
            objective = lambda t: np.sum(continuum_ivar \
                * (continuum_model(t) - continuum_flux)**2)
            op_theta = op.fmin_powell(
                objective, p0_theta, maxiter=np.inf, maxfun=np.inf, disp=False,
                xtol=1e-10, ftol=1e-10)
            """
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            ax.scatter(continuum_disp, continuum_flux,
                alpha=continuum_ivar/np.median(continuum_ivar[continuum_ivar > 0]))
            y = np.dot(op_theta, continuum_matrix)
            ax.plot(continuum_disp, y)

            raise a

            # Evaluate the continuum at all pixels in this region.
            continuum[i, region_mask] *= np.dot(op_theta, region_matrix)
            #continuum2[i, region_mask] = np.dot(unpack(op_theta2), region_matrix)

            # TODO: Store some information about the fit!
            #object_metadata.append()

        metadata.append(tuple(object_metadata))

    if full_output:
        return (continuum, metadata)
    return continuum




    raise a

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(disp, continuum_flux, facecolor="k")
    ax.plot(disp, f(theta), c='r')
    ax.plot(disp, f(moo), c='g')
    ax.plot(dispersion, normalized_flux[0], c='#666666', zorder=-1)

    raise a

    return a

if __name__ == "__main__":

    continuum_pixels = np.loadtxt("pixtest4.txt", dtype=int)



    import os
    from sys import maxsize
    from astropy.table import Table

    # Data.
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

    """
    idx = np.searchsorted(dispersion, 15800)
    dispersion = dispersion[:idx]
    normalized_flux = normalized_flux[:, :idx]
    normalized_ivar = normalized_ivar[:, :idx]
    continuum_pixels = continuum_pixels[continuum_pixels < idx]
    """

    #Let's just do 2000 stars for the moment
    N = 10
    normalized_flux = normalized_flux[:N]
    normalized_ivar = normalized_ivar[:N]

    continuum = fit(dispersion, normalized_flux, normalized_ivar,
        continuum_pixels, regions=regions, order=2, L=1200)

    raise a

    # Find the biggest example.
    mad = np.sum(np.abs(1 - continuum), axis=1)

    indices = (np.argmin(mad), np.argmax(mad))

    # Plot the two distributions of continuum fits?
    for i in indices:
        fig, ax = plt.subplots(2)
        ax[0].scatter(dispersion[continuum_pixels], normalized_flux[i][continuum_pixels], facecolor="k")
        ax[0].plot(dispersion, normalized_flux[i], c="r")
        ax[0].plot(dispersion, continuum[i, :], c="b", zorder=1)
        #ax[1].plot(dispersion, normalized_flux[i], c="r")
        #ax[1].plot(dispersion, normalized_flux[i]/continuum[i], c="b")
        #ax[1].scatter(dispersion[continuum_pixels], normalized_flux[i][continuum_pixels], facecolor="k")
        #ax[1].axhline(1, c="#666666", zorder=-1)
        #ax[0].set_ylim(0.5, 1.1)
        #ax[1].set_ylim(0.5, 1.1)


    raise a

        #axes[1].plot(dispersion, c2[i, :], c="#666666", zorder=-1, alpha=0.5)

    for i in range(N):
        ax.plot(dispersion, normalized_flux[i], c='r', alpha=0.1, zorder=-2)

    raise a




