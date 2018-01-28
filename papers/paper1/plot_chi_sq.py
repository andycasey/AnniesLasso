
"""
Plot the chi-squared distribution for the regularized catalog.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import Table


catalog = Table.read("../tc-cse-regularized-apogee-catalog.fits.gz")

fill_color = "#CCCCCC"

N_bins = 50
xlim = (0, 10)

bins = np.linspace(xlim[0], xlim[1], N_bins)

fig, ax = plt.subplots()
y, bin_edges, _ = ax.hist(catalog["R_CHI_SQ"], bins=bins, facecolor="#666666", edgecolor="k",
    histtype="step", lw=2)

x = np.diff(bin_edges)[0]/2. + bin_edges[:-1]
x2 = np.repeat(x, 2)[1:]
xstep = np.repeat((x[1:] - x[:-1]), 2)
xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
x2 = np.append(x2, x2.max() + xstep[-1])
x2 -= xstep /2.


y2 = np.repeat(y, 2)
ax.fill_between(x2, np.zeros_like(y2), y2, where=np.ones_like(y2),
    color=fill_color,
    zorder=-1)


ax.set_xlabel(r"$\chi_r^2$")
ax.set_ylabel(r"$\rm{Count}$")

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))

ax.set(adjustable="box-forced", aspect=np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

fig.tight_layout()

fig.savefig("test-step-chisq.pdf", dpi=300)
