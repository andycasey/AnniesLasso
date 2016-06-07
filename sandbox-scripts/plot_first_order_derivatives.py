

"""
Plot first-order element coefficients as a function of lambda.
"""


import matplotlib.pyplot as plt
#plt.rcParams["text.usetex"] = True
#plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
import matplotlib.colors as cm
import numpy as np
import os
from scipy import optimize as op
from matplotlib.ticker import MaxNLocator

import colormaps as cmaps

import AnniesLasso as tc


def _show_xlim_changes(fig, diag=0.015, xtol=0):

    N = len(fig.axes)
    if 2 > N: return
    for i, ax in enumerate(fig.axes):

        if i > 0:
            # Put LHS break marks in.
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
            ax.plot((-diag, +diag), (  - diag,   + diag), **kwargs)
            ax.plot((-diag, +diag), (1 - diag, 1 + diag), **kwargs)

        if i != N - 1:
            # Put RHS break marks in.
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
            ax.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), **kwargs) 
            ax.plot((1 - diag, 1 + diag), (  - diag,   + diag), **kwargs)

        # Control spines depending on which axes it is
        if i == 0:
            ax.yaxis.tick_left() 
            ax.spines["right"].set_visible(False)
            ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] - xtol)

        elif i > 0 and i != N - 1:
            ax.set_xlim(ax.get_xlim()[0] + xtol, ax.get_xlim()[1] - xtol)
            ax.yaxis.set_tick_params(size=0)
            ax.tick_params(labelleft='off')
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

        else:
            ax.set_xlim(ax.get_xlim()[0] + xtol, ax.get_xlim()[1])
            ax.yaxis.tick_right()
            ax.tick_params(labelleft='off')
            ax.spines["left"].set_visible(False)

    return None



def plot_first_order_derivatives(model, label_names=None, scaled=True,
    show_clipped_region=False, colors=None, zorders=None,
    clip_less_than=None, label_wavelengths=None, latex_label_names=None,
    wavelength_regions=None, show_legend=True, **kwargs):

    if wavelength_regions is None:
        wavelength_regions = [(model.dispersion[0], model.dispersion[-1])]

    if label_names is None:
        label_names = model.vectorizer.label_names

    if latex_label_names is None:
        latex_label_names = {}

    fig, axes = plt.subplots(1, len(wavelength_regions), figsize=(15, 3.5))

    if len(label_names) > 1:
        #cmap = cm.LinearSegmentedColormap.from_list(
        #    "inferno", cmaps._inferno_data, len(label_names))
        cmap = plt.cm.get_cmap("Set1", len(label_names))
    else:
        cmap = lambda x: "k"

    if colors is not None:
        cmap = lambda x: colors[x % len(colors)]

    axes = np.array(axes).flatten()

    scales = []
    for i, label_name in enumerate(label_names):

        # First order derivatives are always indexed first.
        index = 1 + model.vectorizer.label_names.index(label_name)
        y = model.theta[:, index]
        #y = y
        if clip_less_than is not None:
            y[np.abs(y) < clip_less_than] = 0

        scale = np.nanmax(np.abs(y)) if scaled else 1.
        y = y / scale

        c = cmap(i)
        zorder = 1
        if zorders is not None: zorder = zorders[i]
        for ax in axes:
            ax.plot(model.dispersion, y, c=c, zorder=zorder,
                label=latex_label_names.get(label_name, label_name))

            if clip_less_than is not None and show_clipped_region:
                ax.axhspan(-clip_less_than/scale, +clip_less_than/scale,
                    xmin=-1, xmax=+2,
                    facecolor=c, edgecolor=c, zorder=-100, alpha=0.1)


    # Plot any wavelengths.
    if label_wavelengths is not None:
        label_yvalue = 1.0
        for label_name, wavelengths in label_wavelengths.items():
            try:
                color = cmap(label_names.index(label_name))
                label = None
            except (IndexError, ValueError):
                color = 'k'
                #for ax in axes:
                #    ax.plot([model.dispersion[0] - 1], [0], c=color, label=latex_label_names.get(label_name, label_name))

            for ax in axes:
                ax.plot(wavelengths, label_yvalue * np.ones_like(wavelengths),
                    "|", markersize=20, markeredgewidth=2, c=color)
    
    for ax, wavelength_region in zip(axes, wavelength_regions):
        ax.set_xlim(wavelength_region)
        if scaled:
            ax.set_ylim(-1.2, 1.2)
        
        if ax.is_first_col():
            if scaled:
                ax.set_ylabel(r"$\theta/{\max|\theta|}$")
            else:
                ax.set_ylabel(r"$\theta$")
        else:
            ax.set_yticklabels([])
    
        ax.xaxis.set_major_locator(MaxNLocator(4))

    xlabel = r"$\lambda$ $({\rm\AA})$"
    if len(wavelength_regions) == 1:
        ax.set_xlabel(xlabel)

    else:
        _show_xlim_changes(fig)
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.05, xlabel, rotation='horizontal',
            horizontalalignment='center', verticalalignment='center')

    fig.tight_layout()

    if show_legend:
        axes[0].legend(loc="upper right", ncol=kwargs.get("legend_ncol", len(label_names) % 7), frameon=False)
    fig.subplots_adjust(wspace=0.01, bottom=0.20)

    return fig










if __name__ == "__main__":


    PATH, CATALOG, FILE_FORMAT = ("/Volumes/My Book/surveys/apogee/",
        "apogee-rg.fits", "apogee-rg-custom-normalization-{}.memmap")

    model = tc.load_model("gridsearch-20.0-3.0.model")
    model._dispersion = np.memmap(
        os.path.join(PATH, FILE_FORMAT).format("dispersion"),
        mode="c", dtype=float)



    fig = plot_first_order_derivatives(model,
        label_names=["AL_H", "S_H", "K_H"],
        clip_less_than=np.std(np.abs(model.theta[:, 6])),
        label_wavelengths={
            "AL_H": [16723.524113765838, 16767.938194147067],
            "K_H": [15172.521340566429],
            "Missing": [15235.7, 16755.1]
        },
        latex_label_names={
            "AL_H": r"$[\rm{Al}/\rm{H}]$",
            "K_H": r"$[\rm{K}/\rm{H}]$",
            "S_H": r"$[\rm{S}/\rm{H}]$",
            "Missing": "Missing/Unknown (Shetrone+ 2015)"
        },
        show_clipped_region=True,
        wavelength_regions=[
            (15152.465463818111, 15400),
            (16601, 16800),
        ])
    # Show first figure.
    fig.savefig("papers/sparse-first-order-coefficients.pdf", dpi=300)
    fig.savefig("papers/sparse-first-order-coefficients.png", dpi=300)

    # Now zoom in around those sections.

    colors = []
    cmap = plt.cm.get_cmap("Set1", 3)
    colors = [cmap(0)] + ["#CCCCCC"] * 11 + [cmap(1)] + ["#CCCCCC"] * 2

    fig = plot_first_order_derivatives(model,
        label_names=model.vectorizer.label_names[2:],
        clip_less_than=None, #np.std(np.abs(model.theta[:, 6])),
        scaled=True,
        show_clipped_region=False,
        colors=colors,
        zorders=[10] + [0] * 11 + [10] + [0] * 2,
        show_legend=False,
        latex_label_names={
            "AL_H": r"$[\rm{Al}/\rm{H}]$",
            "CA_H": r"$[\rm{Ca}/\rm{H}]$",
            "C_H": r"$[\rm{C}/\rm{H}]$",
            "FE_H": r"$[\rm{Fe}/\rm{H}]$",
            "K_H": r"$[\rm{K}/\rm{H}]$",
            "MG_H": r"$[\rm{Mg}/\rm{H}]$",
            "MN_H": r"$[\rm{Mn}/\rm{H}]$",
            "NA_H": r"$[\rm{Na}/\rm{H}]$",
            "NI_H": r"$[\rm{Ni}/\rm{H}]$",
            "N_H": r"$[\rm{N}/\rm{H}]$",
            "O_H": r"$[\rm{O}/\rm{H}]$",
            "SI_H": r"$[\rm{Si}/\rm{H}]$",
            "S_H": r"$[\rm{S}/\rm{H}]$",
            "TI_H": r"$[\rm{Ti}/\rm{H}]$",
            "V_H": r"$[\rm{V}/\rm{H}]$"
        },
        label_wavelengths={
            "Missing": [15235.7, 16755.1]
        },
        wavelength_regions=[
            (15235.6 - 10, 10 + 15235.6),
            (16755.1 - 10, 10 + 16755.1)
        ])

    for ax in fig.axes[:-1]:
        ax.set_xticklabels([r"${0:.0f}$".format(_) for _ in ax.get_xticks()])

    fig.savefig("papers/sparse-first-order-coefficients-zoom.pdf", dpi=300)
    fig.savefig("papers/sparse-first-order-coefficients-zoom.png", dpi=300)


