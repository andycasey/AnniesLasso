#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare the APOGEE red clump sample.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import os
from glob import glob
from astropy.io import fits

PATH = "/Users/arc/research/apogee/"
CATALOG = "apogee-rc-DR12.fits"
MEMMAP_FILE_FORMAT = "apogee-rc-{}.memmap"
dtype, ext_flux, ext_sigma = (float, 3, 2)

catalog = fits.open(os.path.join(PATH, CATALOG))[1].data

# Prepare data arrays.
image = fits.open(glob("{path}/{location_id}/*{apogee_id}.fits".format(
    path=PATH, location_id=catalog["LOCATION_ID"][0],
    apogee_id=catalog["APOGEE_ID"][0]))[0])
dispersion = image[ext_flux].header["CRVAL1"] + \
    np.arange(image[ext_flux].data.size) * image[ext_flux].header["CDELT1"]
image.close()

fluxes = np.nan * np.ones((len(catalog), dispersion.size))
flux_uncertainties = np.nan * np.ones_like(fluxes)

failures = []

for i, (location_id, apogee_id) \
in enumerate(zip(catalog["LOCATION_ID"], catalog["APOGEE_ID"])):
    
    spectrum_filename = glob("{path}/{location_id}/*{apogee_id}.fits".format(
        path=PATH, location_id=location_id, apogee_id=apogee_id))[0]

    try:
        image = fits.open(spectrum_filename)
        fluxes[i, :] = image[ext_flux].data
        flux_uncertainties[i, :] = image[ext_sigma].data

    except:
        failures.append(spectrum_filename)

    else:
        image.close()

    print(i, len(catalog))


bad = (flux_uncertainties == 0)
fluxes[bad] = np.nan
flux_uncertainties[bad] = 1.

# Copy to memory-mapped arrays.
memmap_dispersion = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("dispersion"),
    dtype=dtype, mode="w+", shape=dispersion.shape)
memmap_dispersion[:] = dispersion.copy()
memmap_dispersion.flush()
del memmap_dispersion

memmap_fluxes = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("flux"),
    dtype=dtype, mode="w+", shape=fluxes.shape)
memmap_fluxes[:] = fluxes.copy()
memmap_fluxes.flush()
del memmap_fluxes

memmap_flux_uncertainties = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("flux-uncertainties"),
    dtype=dtype, mode="w+", shape=flux_uncertainties.shape)
memmap_flux_uncertainties[:] = flux_uncertainties.copy()
memmap_flux_uncertainties.flush()
del memmap_flux_uncertainties

