
"""
Plot galactic chemical evolution.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import Table

try:
    catalog
except NameError:
    catalog = Table.read("../tc-cse-regularized-apogee-catalog.fits.gz")
    ok = catalog["OK"] * (catalog["R_CHI_SQ"] < 3) * (catalog["TEFF"] > 4000) * (catalog["TEFF"] < 5500)

else:
    print("USING PRE-LOADED CATALOG")

log_scale = True
gridsize = 100
elements =  [
 'Al',
 'Ca',
 'C',
 'K',
 'Mg',
 'Mn',
 'Na',
 'Ni',
 'N',
 'O',
 'Si',
 'S',
 'Ti',
 'V']

fig, axes = plt.subplots(int(len(elements)/2.), 2, figsize=(9, 12))
axes = np.array(axes).flatten()

extents = {
    None: (-1, +1),
    "Al": (-0.6, 0.6),
    "Ca": (-0.6, 0.6),
    "Mg": (-0.6, 0.6),
    "Mn": (-0.6, 0.6),
    "Si": (-0.6, 0.6),
    "S": (-0.6, 0.6)
}


for i, (ax, element) in enumerate(zip(axes, elements)):

    x = catalog["FE_H"]
    y = catalog["{}_H".format(element.upper())] - catalog["FE_H"]

    ylims = extents.get(element, extents[None])
    ax.hexbin(x[ok], y[ok], cmap="viridis", bins="log" if log_scale else None,
        gridsize=gridsize,
        extent=(-2, 0.5, ylims[0], ylims[1]), mincnt=None,
        rasterized=True)

    ax.set_xlim(-2, 0.50)
    ax.set_ylim(ylims)

    ax.set_ylabel(r"$[\rm{{{0}}}/\rm{{Fe}}]$".format(element.title()))
    if ax.is_last_row():
        ax.set_xlabel(r"$[\rm{Fe}/\rm{H}]$")
    else:
        ax.set_xticklabels([])

    if element not in extents:
        ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
    else:
        ax.set_yticks([-0.5, 0, 0.5])

    # Show scaled-solar line.
    ax.axhline(0, c="#FFFFFF", alpha=0.5, linestyle=":")
    ax.axvline(0, c="#FFFFFF", alpha=0.5, linestyle=":")

    # Show a representative error bar.
    xerr = np.mean(catalog["E_FE_H"])
    yerr = np.mean(np.sqrt(
        catalog["E_FE_H"]**2 + catalog["E_{}_H".format(element.upper())]**2))

    xpos = ax.get_xlim()[0] + 0.95 * np.ptp(ax.get_xlim())
    ypos = ax.get_ylim()[0] + 0.05 * np.ptp(ax.get_ylim())

    #ax.errorbar([xpos], [ypos], xerr=[xerr], yerr=[yerr], fmt=None,
    #    ecolor="#FFFFFF", zorder=10, lw=2)

fig.tight_layout()
fig.savefig("test-step-gce.pdf", dpi=300)


