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
dtype, ext_flux, ext_sigma = (float, 1, 2)

catalog = fits.open(os.path.join(PATH, CATALOG))[1].data

# Prepare data arrays.
image = fits.open(glob("{path}/{location_id}/*{apogee_id}.fits".format(
    path=PATH, location_id=catalog["LOCATION_ID"][0],
    apogee_id=catalog["APOGEE_ID"][0]))[0])
dispersion = 10**(image[ext_flux].header["CRVAL1"] + \
    np.arange(image[ext_flux].data.size) * image[ext_flux].header["CDELT1"])
image.close()

normalized_flux = np.nan * np.ones((len(catalog), dispersion.size))
normalized_ivar = np.nan * np.ones_like(normalized_flux)

failures = []

for i, (location_id, apogee_id) \
in enumerate(zip(catalog["LOCATION_ID"], catalog["APOGEE_ID"])):
    
    spectrum_filename = glob("{path}/{location_id}/*{apogee_id}.fits".format(
        path=PATH, location_id=location_id, apogee_id=apogee_id))[0]

    try:
        image = fits.open(spectrum_filename)
        normalized_flux[i, :] = image[ext_flux].data
        normalized_ivar[i, :] = 1.0/(image[ext_sigma].data**2)

    except:
        failures.append(spectrum_filename)

    else:
        image.close()

    print(i, len(catalog))

bad = ~np.isfinite(normalized_ivar)
normalized_ivar[bad] = 0.
normalized_flux[bad] = 1.

# Copy to memory-mapped arrays.
memmap_dispersion = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("dispersion"),
    dtype=dtype, mode="w+", shape=dispersion.shape)
memmap_dispersion[:] = dispersion.copy()
memmap_dispersion.flush()
del memmap_dispersion

memmap_normalized_flux = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-flux"),
    dtype=dtype, mode="w+", shape=normalized_flux.shape)
memmap_normalized_flux[:] = normalized_flux.copy()
memmap_normalized_flux.flush()
del memmap_normalized_flux

memmap_normalized_ivar = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-ivar"),
    dtype=dtype, mode="w+", shape=normalized_ivar.shape)
memmap_normalized_ivar[:] = normalized_ivar.copy()
memmap_normalized_ivar.flush()
del memmap_normalized_ivar

