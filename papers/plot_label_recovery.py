"""
Plot the label recovery for the validation set as a function of S/N.
"""

import numpy as np

from matplotlib.ticker import MaxNLocator
from six.moves import cPickle as pickle
from collections import Counter, OrderedDict

models = {
    "model1": "gridsearch-2.0-3.0-s2-heuristically-set.model.individual_visits"
}


label_names = OrderedDict([
    ('TEFF', r'$T_{\rm eff}$ $(\rm{K})$'),
    ('LOGG', r'$\log{g}$ $(\rm{dex})$'),
    ('AL_H', r'$[\rm{Al}/\rm{H}]$ $(\rm{dex})$'),
    ('CA_H', r'$[\rm{Ca}/\rm{H}]$ $(\rm{dex})$'),
    ('C_H', r'$[\rm{C}/\rm{H}]$ $(\rm{dex})$'),
    ('FE_H', r'$[\rm{Fe}/\rm{H}]$ $(\rm{dex})$'),
    ('K_H', r'$[\rm{K}/\rm{H}]$ $(\rm{dex})$'),
    ('MG_H', r'$[\rm{Mg}/\rm{H}]$ $(\rm{dex})$'),
    ('MN_H', r'$[\rm{Mn}/\rm{H}]$ $(\rm{dex})$'),
    ('NA_H', r'$[\rm{Na}/\rm{H}]$ $(\rm{dex})$'),
    ('NI_H', r'$[\rm{Ni}/\rm{H}]$ $(\rm{dex})$'),
    ('N_H', r'$[\rm{N}/\rm{H}]$ $(\rm{dex})$'),
    ('O_H', r'$[\rm{O}/\rm{H}]$ $(\rm{dex})$'),
    ('SI_H', r'$[\rm{Si}/\rm{H}]$ $(\rm{dex})$'),
    ('S_H', r'$[\rm{S}/\rm{H}]$ $(\rm{dex})$'),
    ('TI_H', r'$[\rm{Ti}/\rm{H}]$ $(\rm{dex})$'),
    ('V_H', r'$[\rm{V}/\rm{H}]$ $(\rm{dex})$'),
])


# Load the APOGEE IDs from individual visits from a separate file because I am
# an idiot.
with open("apogee_ids_from_individual_visits.pkl", "rb") as fp:
    apogee_ids = pickle.load(fp)

fig, axes = plt.subplots(6, 3, figsize=(8.5, 12.5))
axes = np.array(axes).flatten()

N_repeats_min = 2
N_bins = 25
colors = "kr"
wrt_cannon = True
common_ylim = 0.5
minimum_apogee_snr = 50.0 # Mark the minimum S/N in any combined APOGEE spectra


counts = Counter(apogee_ids)
is_repeated_enough = np.array(
    [(counts[_] >= N_repeats_min) for _ in apogee_ids])


# Define the metric to show at each bin.
# MAD
metric = lambda differences: np.median(np.abs(differences))

# RMS
#metric = lambda differences: np.sqrt(np.sum((differences**2))/differences.size)


for i, (model_name, model_filename) in enumerate(models.items()):

    color = colors[i]
    fill_color = "#CCCCCC"

    # Load the data:
    #   (snrs, high_snr_expected, high_snr_inferred, differences_expected, 
    #    differences_inferred, single_visit_inferred)
    with open(model_filename, "rb") as fp:
        contents = pickle.load(fp)

    # Let's unpack that.
    iv_snr, combined_aspcap, combined_cannon, \
        differences_aspcap, differences_cannon, \
        iv_cannon = contents


    # Which label residuals should we show?
    differences = differences_cannon if wrt_cannon else differences_aspcap

    _, bin_edges, __ = axes[0].hist(iv_snr[is_repeated_enough], 
        histtype="step", lw=2, color=color, bins=N_bins)
    axes[0].hist(iv_snr[is_repeated_enough], bins=N_bins,
        facecolor=fill_color, edgecolor=fill_color, zorder=-1)
    axes[0].set_ylabel(r"$\rm{Count}$")

    #axes[0].text(0.95, 0.95, r"${0:.0f}$".format(is_repeated_enough.sum()),
    #    color=color, transform=axes[0].transAxes,
    #    horizontalalignment="right", verticalalignment="bottom")


    for j, label_name in enumerate(label_names.keys()):

        # Take the mean per bin.
        x = np.diff(bin_edges)[0]/2. + bin_edges[:-1]

        y = []
        for k in range(len(bin_edges) - 1):
            mask = (bin_edges[k + 1] > iv_snr) * (iv_snr >= bin_edges[k]) \
                 * is_repeated_enough
            if not any(mask):
                y.append(np.nan)
            else:
                y.append(metric(differences_cannon[mask, j]))

        y = np.array(y)

        ax = axes[j + 1]
        ax.plot(x, y, drawstyle="steps-mid", c=color, lw=2)

        # fuck
        x2 = np.repeat(x, 2)[1:]
        xstep = np.repeat((x[1:] - x[:-1]), 2)
        xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
        x2 = np.append(x2, x2.max() + xstep[-1])
        x2 -= xstep /2.

        y2 = np.repeat(y, 2)
        ax.fill_between(x2, np.zeros_like(y2), y2, where=np.ones_like(y2), 
            color=fill_color, 
            zorder=-1)
        ax.set_ylabel(label_names[label_name])



# Set common y-axes for abundance labels?
ylim = common_ylim  if common_ylim is not None \
                    else np.max([ax.get_ylim()[1] for ax in axes[3:]])

# Some custom limits:
# MAGIC
axes[1].set_ylim(0, 100)
axes[2].set_ylim(0, 0.2)

axes[0].yaxis.set_major_locator(MaxNLocator(4))
axes[1].yaxis.set_major_locator(MaxNLocator(4))
axes[2].set_yticks([0, 0.1, 0.2])

for ax in axes[3:]:   
    ax.set_ylim(0, ylim)
    ax.set_yticks([0, 0.2, 0.4])



for ax in axes:

    if minimum_apogee_snr is not None:
        ax.axvline(minimum_apogee_snr, c="#666666", linestyle="--", zorder=-1)

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    
    if not ax.is_last_row():
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(r"$S/N$")

    ax.set(adjustable="box-forced", aspect=np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

fig.tight_layout()

fig.savefig("papers/validation-label-recovery.pdf", dpi=300)

raise a