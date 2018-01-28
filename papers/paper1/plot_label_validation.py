"""
Plot the labels for the validation set.

"""

import numpy as np

from matplotlib.ticker import MaxNLocator
from six.moves import cPickle as pickle
from collections import Counter, OrderedDict



validation_filename = "../gridsearch-2.0-3.0-s2-heuristically-set.model.validation"


with open(validation_filename, "rb") as fp:
    expected, inferred, cov, metadata = pickle.load(fp)



label_names = OrderedDict([
    ('TEFF', r'$T_{\rm eff}$'),
    ('LOGG', r'$\log{g}$'),
    ('AL_H', r'$[\rm{Al}/\rm{H}]$'),
    ('CA_H', r'$[\rm{Ca}/\rm{H}]$'),
    ('C_H', r'$[\rm{C}/\rm{H}]$'),
    ('FE_H', r'$[\rm{Fe}/\rm{H}]$'),
    ('K_H', r'$[\rm{K}/\rm{H}]$'),
    ('MG_H', r'$[\rm{Mg}/\rm{H}]$'),
    ('MN_H', r'$[\rm{Mn}/\rm{H}]$'),
    ('NA_H', r'$[\rm{Na}/\rm{H}]$'),
    ('NI_H', r'$[\rm{Ni}/\rm{H}]$'),
    ('N_H', r'$[\rm{N}/\rm{H}]$'),
    ('O_H', r'$[\rm{O}/\rm{H}]$'),
    ('SI_H', r'$[\rm{Si}/\rm{H}]$'),
    ('S_H', r'$[\rm{S}/\rm{H}]$'),
    ('TI_H', r'$[\rm{Ti}/\rm{H}]$'),
    ('V_H', r'$[\rm{V}/\rm{H}]$'),
])





fig, axes = plt.subplots(6, 3, figsize=(8.5, 12.5))
axes = np.array(axes).flatten()


for i, (ax, label_name) in enumerate(zip(axes, label_names.keys())):
    differences = inferred[:, i] - expected[:, i]
    ax.scatter(expected[:, i], inferred[:, i], marker="+", alpha=0.5,
        c="k", s=30, lw=1.5)

    mean = np.mean(differences)
    sigma = np.std(differences)

    if np.round(mean, 2) == 0.00: mean = 0
    if np.round(sigma, 2) == 0.00: sigma = 0

    format = r"${0:.0f}$" if label_name == "TEFF" else r"${0:.2f}$"

    ax.text(0.90, 0.23, format.format(mean), color="k",
        horizontalalignment="right", verticalalignment="bottom",
        transform=ax.transAxes)
    ax.text(0.90, 0.10, format.format(sigma), color="k",
        horizontalalignment="right", verticalalignment="bottom",
        transform=ax.transAxes)

    #ax.set_xlabel(label_names.get(label_name, label_name) + r" $({\rm ASPCAP})$")
    #ax.set_ylabel(label_names.get(label_name, label_name) + r" $({\rm Cannon})$")
    #ax.set_title(label_names.get(label_name, label_name))

    ax.text(0.10, 0.90, label_names.get(label_name, label_name),
        color="k", horizontalalignment="left", verticalalignment="top",
        transform=ax.transAxes)



# Common limits for all axes.
limits = np.array([[axes[0].get_xlim(), axes[0].get_ylim()]]).flatten()
limits = (min(limits), max(limits))
axes[0].set_xlim(limits)
axes[0].set_ylim(limits)
axes[0].plot(limits, limits, c="#666666", ls="--", zorder=-1)
axes[0].set(adjustable="box-forced", aspect=np.ptp(axes[0].get_xlim())/np.ptp(axes[0].get_ylim()))
axes[0].xaxis.set_major_locator(MaxNLocator(3))
axes[0].yaxis.set_major_locator(MaxNLocator(3))



limits = np.array([[axes[1].get_xlim(), axes[1].get_ylim()]]).flatten()
limits = (min(limits), max(limits))
axes[1].set_xlim(limits)
axes[1].set_ylim(limits)
axes[1].plot(limits, limits, c="#666666", ls="--", zorder=-1)
axes[1].set(adjustable="box-forced", aspect=np.ptp(axes[1].get_xlim())/np.ptp(axes[1].get_ylim()))
axes[1].xaxis.set_major_locator(MaxNLocator(3))
axes[1].yaxis.set_major_locator(MaxNLocator(3))


limits = (-3, 1)
for ax in axes[2:]:
    ax.plot(limits, limits, c="#666666", ls="--", zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set(adjustable="box-forced", aspect=np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    if not ax.is_last_row():
        ax.set_xticklabels([])

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))

fig.tight_layout()
axes[-1].set_visible(False)

fig.savefig("regularized-model-validation.pdf", dpi=300)

