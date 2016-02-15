"""
Plot a Hertzsprung-Russell diagram of the stars in the training set,
and a histogram of the [X/H] abundances for each element.
"""

import numpy as np
from astropy.table import Table

import matplotlib.colors as cm
import matplotlib.pyplot as plt
from colormaps import viridis, _viridis_data
from matplotlib.ticker import MaxNLocator

labelled_set = Table.read("/Users/arc/research/apogee/apogee-rg.fits")

# Identify the training set
np.random.seed(123)
q = np.random.randint(0, 10, len(labelled_set)) % 10
training_set = labelled_set[(q > 0)]


elements = [label for label in labelled_set.dtype.names \
    if label.endswith("_H") and label not in ("PARAM_M_H", "SRC_H")]

# Get the bin sizes.
abundances = np.array([training_set[element] for element in elements])


fig, (ax_hrd, ax_hist) = plt.subplots(1, 2, figsize=(12.6, 6.3))

hrd = ax_hrd.scatter(training_set["TEFF"], training_set["LOGG"],
    c=training_set["FE_H"], cmap=viridis, s=50)

ax_hrd.set_xlim(ax_hrd.get_xlim()[::-1])
ax_hrd.set_ylim(4.5, -0.5)

ax_hrd.set_xlabel(r"$T_{\rm eff}$ $(K)$")
ax_hrd.set_ylabel(r"$\log{g}$")

ax_hrd.xaxis.set_major_locator(MaxNLocator(5))
ax_hrd.yaxis.set_major_locator(MaxNLocator(5))


bins = np.linspace(abundances.min(), abundances.max(), 25)

cmap = cm.LinearSegmentedColormap.from_list(
    "viridis", _viridis_data, len(elements))

for j, element in enumerate(elements):
    ax_hist.hist(training_set[element], bins=bins, color=cmap(j),
        histtype="step", lw=2, label=None, normed=True)

    ax_hist.plot([], [], color=cmap(j), lw=2, 
        label=r"${\rm " + element.split("_")[0].title() + r"}$")

ax_hist.legend(frameon=False, loc="upper left", ncol=2)
ax_hist.set_xlabel(r"$[{\rm X}/{\rm H}]$")
ax_hist.yaxis.tick_right()
ax_hist.yaxis.set_label_position("right")
ax_hist.yaxis.set_major_locator(MaxNLocator(5))
ax_hist.xaxis.set_major_locator(MaxNLocator(5))

ax_hist.set_ylabel(r"${\rm Normalized}$ ${\rm count}$")


ax_hrd.set(adjustable='box-forced', aspect=np.ptp(ax_hrd.get_xlim())/np.ptp(ax_hrd.get_ylim()))
ax_hist.set(adjustable='box-forced', aspect=np.ptp(ax_hist.get_xlim())/np.ptp(ax_hist.get_ylim()))
fig.tight_layout()


pos = ax_hrd.get_position()

ax_hrd_cbar = fig.add_axes([pos.x0 + 0.025, pos.y0 + pos.height - 0.125, 
    pos.width / 2.0, 0.05])
cbar_hrd = plt.colorbar(hrd, cax=ax_hrd_cbar, orientation="horizontal")

cbar_hrd.set_ticks([-2, -1.5, -1, -0.5, 0, 0.5])
cbar_hrd.ax.set_xlabel(r"$[{\rm Fe}/{\rm H}]$")


fig.savefig("training_set_hrd.pdf", dpi=300)



#     cmaps._viridis_data,
#len(scale_factors))