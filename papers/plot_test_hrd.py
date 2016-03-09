
"""
Plot the extent of the bonafide giant sample.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import Table


import matplotlib.colors as cm
from colormaps import viridis, _viridis_data

try:
    catalog
except NameError:
    catalog = Table.read("../tc-cse-regularized-apogee-catalog.fits.gz")
    ok = catalog["OK"] * (catalog["R_CHI_SQ"] < 3) * (catalog["TEFF"] > 4000) * (catalog["TEFF"] < 5500)

else:
    print("USING PRE-LOADED CATALOG")

test_set = catalog[ok]

elements = [label for label in test_set.dtype.names \
    if  label.endswith("_H")  and \
        label.count("_") == 1 and \
        label not in ("PARAM_M_H", "SRC_H")]

# Get the bin sizes.
abundances = np.array([test_set[element] for element in elements])


fig, (ax_hrd, ax_hexbin, ax_hist) = plt.subplots(1, 3, figsize=(18.6, 6.3))

sort_by = np.argsort(test_set["FE_H"])
hrd = ax_hrd.scatter(test_set["TEFF"][sort_by], test_set["LOGG"][sort_by],
    c=test_set["FE_H"][sort_by], cmap=viridis, s=50, rasterized=True,
    lw=0.25, vmin=-2, vmax=0.5)

ax_hrd.set_xlim(5750, 3750)
ax_hrd.set_ylim(4.5, -0.5)

ax_hrd.set_xlabel(r"$T_{\rm eff}$ $(K)$")
ax_hrd.set_ylabel(r"$\log{g}$")

ax_hrd.xaxis.set_major_locator(MaxNLocator(5))
ax_hrd.yaxis.set_major_locator(MaxNLocator(5))

ax_hexbin.hexbin(test_set["TEFF"], test_set["LOGG"], bins="log",
    cmap="viridis", gridsize=75, marginals=True,
    extent=(ax_hrd.get_xlim()[0], ax_hrd.get_xlim()[1], ax_hrd.get_ylim()[0],
        ax_hrd.get_ylim()[1]))

ax_hexbin.set_xlim(ax_hrd.get_xlim())
ax_hexbin.set_ylim(ax_hrd.get_ylim())

ax_hexbin.set_xlabel(ax_hrd.get_xlabel())
ax_hexbin.set_ylabel(ax_hrd.get_ylabel())

ax_hexbin.xaxis.set_major_locator(MaxNLocator(5))
ax_hexbin.yaxis.set_major_locator(MaxNLocator(5))

bins = np.linspace(-3, 1, 25)

cmap = cm.LinearSegmentedColormap.from_list(
    "viridis", _viridis_data, len(elements))

for j, element in enumerate(elements):
    ax_hist.hist(test_set[element], bins=bins, color=cmap(j),
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
ax_hexbin.set(adjustable='box-forced', aspect=np.ptp(ax_hrd.get_xlim())/np.ptp(ax_hrd.get_ylim()))
ax_hist.set(adjustable='box-forced', aspect=np.ptp(ax_hist.get_xlim())/np.ptp(ax_hist.get_ylim()))


fig.tight_layout()


pos = ax_hrd.get_position()

ax_hrd_cbar = fig.add_axes([pos.x0 + 0.020, pos.y0 + pos.height - 0.130, 
    pos.width * 0.60, 0.05])
cbar_hrd = plt.colorbar(hrd, cax=ax_hrd_cbar, orientation="horizontal")

cbar_hrd.set_ticks([-2, -1.5, -1, -0.5, 0, 0.5])
cbar_hrd.ax.set_xlabel(r"$[{\rm Fe}/{\rm H}]$")


fig.savefig("test_set_hrd.pdf", dpi=300)
