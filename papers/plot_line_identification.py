

"""
Plot the model coefficients to illustrate line identification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import AnniesLasso as tc
import matplotlib.colors as cm

from matplotlib.ticker import MaxNLocator
import colormaps as cmaps


model = tc.load_model("../gridsearch-2.0-3.0-s2-heuristically-set.model.ignore")

PATH, FILE_FORMAT = (
    "/Users/arc/research/apogee",
    "apogee-rg-custom-normalization-{}.memmap")
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)



display_limits = {
    "AL_H": 0.3,
    "S_H": 0.2,
    "K_H": 0.2
}
terms = ["AL_H", "S_H", "K_H"]

colors = ["#24808E", "#1EA384", "#471365"]
colors = ["#FA9406", "#C0394F", "#230B4D"]
wavelength_regions = [
    (15120, 15250),
    (16670, 16800)
]

known_lines = {
    "AL_H": [16723.524113765838, 16767.938194147067],
    "K_H": [15167.21089680259, 15172.521340566429]
}

latex_labels = {
    "AL_H": r"$[\rm{Al}/\rm{H}]$",
    "K_H": r"$[\rm{K}/\rm{H}]$",
    "S_H": r"$[\rm{S}/\rm{H}]$"
}

mask = np.zeros(dispersion.size, dtype=bool)
for start, end in wavelength_regions:
    mask[(end >= dispersion) * (dispersion >= start)] = True


fig, axes = plt.subplots(1, len(wavelength_regions),
    figsize=(11.275, 3.25))
hrlv = model.vectorizer.get_human_readable_label_vector()


for i, label in enumerate(model.vectorizer.label_names[2:]):
    if label in terms:
        c = colors[terms.index(label)]
    else:
        c = "#666666"

    #for ax in axes:
    #    ax.plot(dispersion, model.theta[:, i + 3], c=c, zorder=-1)


for term, color in zip(terms, colors):

    term_index = hrlv.index(term)

    for ax in axes:
        y = model.theta[:, term_index]
        y = y / np.max(np.abs(model.theta[:, term_index]))

        y[np.abs(y) < display_limits.get(term)] = 0


        ax.plot(dispersion, y, c=color, lw=2, label=latex_labels.get(term, term))
        #ax.axhspan(-np.std(y), +np.std(y), facecolor=color, edgecolor=color, 
        #    alpha=0.5, zorder=-1)

for ax, wavelength_region in zip(axes, wavelength_regions):
    ax.set_xlim(*wavelength_region)
    ax.set_ylim(-1.1, 1.1)

    if ax.is_first_col():
        ax.set_ylabel(r"$\theta/\max|\theta|$")

    ax.xaxis.set_major_locator(MaxNLocator(4))



diag = 0.015
for ax in axes:

    for element, wavelengths in known_lines.items():

        color = colors[terms.index(element)]

        for wavelength in wavelengths:
            ax.plot([wavelength, wavelength],
                [0.85, 1.00], lw=3, c=color)

            ax.axvline(wavelength, lw=1, c=color, linestyle=":", zorder=-1)



    if ax.is_last_col():

        # Put LHS break marks in.
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax.plot((-diag, +diag), (  - diag,   + diag), **kwargs)
        ax.plot((-diag, +diag), (1 - diag, 1 + diag), **kwargs)


    else:
        # Put RHS break marks in.
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), **kwargs) 
        ax.plot((1 - diag, 1 + diag), (  - diag,   + diag), **kwargs)

    # Control spines depending on which axes it is
    if ax.is_first_col():
        ax.yaxis.tick_left() 
        ax.spines["right"].set_visible(False)

    else:
        ax.yaxis.tick_right()
        ax.set_yticklabels([])
        ax.tick_params(labelleft='off')
        ax.spines["left"].set_visible(False)

    ax.set(adjustable="box-forced", aspect=0.5 * np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

ax.set_xlabel("empty")

fig.tight_layout()
fig.subplots_adjust(wspace=0.01, bottom=0.2)

axes[0].set_xticks(axes[0].get_xticks()[:-1])
axes[1].set_xticks(axes[1].get_xticks()[1:])

ax.set_xlabel("")

lax = fig.add_subplot(1, 1, 1)
lax.set(adjustable="box-forced", aspect=np.ptp(lax.get_xlim())/np.ptp(lax.get_ylim()))
lax.set_frame_on(False)
lax.set_xticks([])
lax.set_yticks([])
lax.set_xlabel(r"$\rm{Wavelength},$ $\lambda$ $(\rm{\AA})$", labelpad=20)

axes[0].legend(loc="upper left", frameon=False, fontsize=14)



fig.savefig("line-identification.pdf", dpi=300)


fig, axes = plt.subplots(1, len(wavelength_regions),
    figsize=(11.275, 3.25))


for i, label in enumerate(model.vectorizer.label_names[2:]):
    if label in terms:
        c, lw, zorder = colors[terms.index(label)], 2, 10
    else:
        c, lw, zorder = "#CCCCCC", 1, 1

    for ax in axes:
        ax.plot(dispersion, model.theta[:, i + 3]/np.std(model.theta[:, i + 3]), lw=lw, c=c, zorder=zorder)



wavelengths = [15235.8, 16755.6]

for ax, wavelength in zip(axes, wavelengths):

    ax.set_xlim(wavelength - 5, wavelength + 5)
    ax.set_ylim(-15, 5)


    if ax.is_last_col():

        # Put LHS break marks in.
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax.plot((-diag, +diag), (  - diag,   + diag), **kwargs)
        ax.plot((-diag, +diag), (1 - diag, 1 + diag), **kwargs)


    else:
        # Put RHS break marks in.
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
        ax.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), **kwargs) 
        ax.plot((1 - diag, 1 + diag), (  - diag,   + diag), **kwargs)

    # Control spines depending on which axes it is
    if ax.is_first_col():
        ax.yaxis.tick_left() 
        ax.spines["right"].set_visible(False)

    else:
        ax.yaxis.tick_right()
        ax.set_yticklabels([])
        ax.tick_params(labelleft='off')
        ax.spines["left"].set_visible(False)

    ax.set(adjustable="box-forced", aspect=0.5 * np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels(["{0:.0f}".format(each) for each in ax.get_xticks()])

fig.tight_layout()
fig.subplots_adjust(wspace=-0.11, bottom=0.2)

axes[0].set_ylabel(r"$\theta/\sigma_\theta$")

lax = fig.add_subplot(1, 1, 1)
lax.set(adjustable="box-forced", aspect=np.ptp(lax.get_xlim())/np.ptp(lax.get_ylim()))
lax.set_frame_on(False)
lax.set_xticks([])
lax.set_yticks([])
lax.set_xlabel(r"$\rm{Wavelength},$ $\lambda$ $(\rm{\AA})$", labelpad=20)


fig.savefig("line-identification-zoom.pdf", dpi=300)



#ax.set_xlabel("Test")