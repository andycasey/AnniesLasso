#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download the APOGEE red giant branch training sample (the apStar files).
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
from astropy.io import fits

PATH = "/Users/arc/research/apogee/"
CATALOG = "apogee-rg.fits"
catalog = fits.open(os.path.join(PATH, CATALOG))[1].data


wget_command = (
    "wget -O apogee-rg-apStar/{location_id}/apStar-r5-{apogee_id}.fits "
    "http://data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/stars/apo25m/"
    "{location_id}/apStar-r5-{apogee_id}.fits")

N = len(catalog)
for i, (location_id, apogee_id) \
in enumerate(zip(catalog["LOCATION_ID"], catalog["APOGEE_ID"])):

    kwds = {
        "location_id": location_id,
        "apogee_id": apogee_id
    }
    if not os.path.exists("apogee-rg-apStar/{location_id}".format(**kwds)):
        os.mkdir("apogee-rg-apStar/{location_id}".format(**kwds))

    # Download the file.
    filename = "apogee-rg-apStar/{location_id}/apStar-r5-{apogee_id}.fits"\
        .format(**kwds)

    if not os.path.exists(filename):
        os.system(wget_command.format(**kwds))

    print(i, N)



