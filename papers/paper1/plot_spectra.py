"""
Plot some spectra and model spectra -- show that we are better than ASPCAP.
"""



import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
from matplotlib.ticker import MaxNLocator

import colormaps as cmaps
import AnniesLasso as tc


# Need all data.
np.random.seed(123) # For reproducibility.

# Data.
PATH, CATALOG, FILE_FORMAT = ("/Users/arc/research/apogee", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)


# Split up the data into ten random subsets.
np.random.seed(123)
q = np.random.randint(0, 10, len(labelled_set)) % 10

validate_set = (q == 0)



# OK,
model = tc.load_model("../gridsearch-2.0-3.0-s2-heuristically-set.model")

wavelengths = (15400, 15600)


index = np.random.randint(0, 1460)

#fig, axes = plt.subplots(2, 1, figsize=(13.3, 4))
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 1,
                       height_ratios=[1,4]
                       )

fig = plt.figure(figsize=(13.3, 4))
ax_residual = plt.subplot(gs[0])
ax_spectrum = plt.subplot(gs[1])


ax_spectrum.plot(dispersion, normalized_flux[validate_set][index],
    drawstyle='steps-mid', c='k')

    
aspcap_color, tc_color = ("#3399CC", "r")


# fuck
x2 = np.repeat(dispersion, 2)[1:]
xstep = np.repeat((dispersion[1:] - dispersion[:-1]), 2)
xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
x2 = np.append(x2, x2.max() + xstep[-1])
x2 -= xstep /2.

sigma = 1.0/(normalized_ivar[validate_set][index]**0.5)
y_low = np.repeat(normalized_flux[validate_set][index] - sigma, 2)
y_high = np.repeat(normalized_flux[validate_set][index] + sigma, 2)

ax_spectrum.fill_between(x2, y_low, y_high, where=np.ones_like(y_low), 
    color="#CCCCCC", zorder=-1)
ax_spectrum.plot(x2, y_low, c="#BBBBBB")
ax_spectrum.plot(x2, y_high, c="#BBBBBB")


# Show the ASPCAP best-fit model.
filename = "aspcapStar-r5-v603-{}.fits".format(labelled_set["APOGEE_ID"][validate_set][index])
if os.path.exists(filename):

    image = fits.open(filename)
    ax_spectrum.plot(dispersion, image[-2].data, c=aspcap_color, lw=1.5,
        label=r"${\rm ASPCAP}$")


    residuals = image[-2].data - normalized_flux[validate_set][index]

    #ax_residual.plot(dispersion, residuals, c=aspcap_color, drawstyle="steps-mid")
    r2 = np.repeat(residuals, 2)
    ax_residual.fill_between(x2, 0, r2, facecolor=aspcap_color, edgecolor=aspcap_color)

else:
    print("No ASPCAP model shown. Missing {}".format(filename))


# Fit it.
fitted_labels = model.fit(normalized_flux[validate_set][index],
    normalized_ivar[validate_set][index])

model_spectra = model.predict(fitted_labels.flatten()).flatten()
ax_spectrum.plot(dispersion, model_spectra, c=tc_color,
    label=r"${\rm The}$ ${\rm Cannon}$")# drawstyle='steps-mid')



residuals = model_spectra - normalized_flux[validate_set][index]

ax_residual.axhline(0, c="#666666", linestyle=":", zorder=-1)
#ax_residual.plot(dispersion, residuals, c=tc_color, drawstyle='steps-mid')
r2 = np.repeat(residuals, 2)
ax_residual.fill_between(x2, 0, r2, facecolor=tc_color, zorder=10, edgecolor=tc_color)


ax_spectrum.legend(loc="lower left", frameon=False)
ax_spectrum.text(0.98, 0.10, r"${{\rm {0}}}$".format(
    labelled_set["APOGEE_ID"][validate_set][index]), color="k",
    horizontalalignment="right", verticalalignment="bottom",
    transform=ax_spectrum.transAxes)

ax_spectrum.set_xlim(wavelengths)
ax_spectrum.set_ylim(0.5, 1.1)

ax_residual.set_xlim(ax_spectrum.get_xlim())
ax_residual.set_xticklabels([])
ax_residual.set_ylim(-0.05, 0.05)

ax_residual.yaxis.set_major_locator(MaxNLocator(3))
ax_residual.xaxis.set_major_locator(MaxNLocator(6))
ax_spectrum.xaxis.set_major_locator(MaxNLocator(6))
ax_spectrum.yaxis.set_major_locator(MaxNLocator(4))

ax_spectrum.set_ylabel(r"${\rm Normalized}$ ${\rm flux}$")
ax_spectrum.set_xlabel(r"${\rm Wavelength,}$ $\lambda$ $({\rm \AA})$")

#ax_residual.set_ylabel(r"$\Delta$")

# Set aspect of the residuals plot.
#ax_residual.set(adjustable="box-forced", aspect=2.0 * np.ptp(ax_residual.get_xlim())/np.ptp(ax_residual.get_ylim()))

fig.tight_layout()



fig.savefig("spectrum.pdf", dpi=300)


