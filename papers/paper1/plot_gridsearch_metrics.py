"""
Plot some quality metrics for gridsearch models.
"""

import matplotlib
matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import logging

from glob import glob
from six.moves import cPickle as pickle



def plot_test_scalar_metrics(metric_function, filenames, metric_label=None,
    debug=True, **kwargs):

    Lambdas = []
    scale_factors = []
    metrics = []

    for filename in filenames:

        try:
            _ = filename.split("-")
            scale_factor = float(_[1])
            log10_Lambda = float(_[2].split(".model")[0])

        except:
            print("Skipping filename {}".format(filename))
            continue

        with open(filename, "rb") as fp:
            contents = pickle.load(fp, encoding="latin-1")

        snrs, high_snr_expected, high_snr_inferred, \
            differences_expected, differences_inferred, \
            single_visit_inferred = contents

        # Calculate the metric.
        try:
            metric = metric_function(*contents)
            metric = float(metric) # Must be a float

        except:
            logging.exception("Failed to calculate metric for {}".format(filename))
            if debug: raise

        else:

            metrics.append(metric)
            Lambdas.append(10**log10_Lambda)
            scale_factors.append(scale_factor)

    metrics = np.array(metrics)
    Lambdas = np.array(Lambdas)
    scale_factors = np.array(scale_factors)

    # Make the figure
    fig, ax = plt.subplots()

    # scale factors are non-linear, so lets show them as indices then we will
    # adjust the y-ticks and labels as necessary
    unique_scale_factors = list(np.sort(np.unique(scale_factors)))
    scale_factor_indices \
        = np.array([unique_scale_factors.index(_) for _ in scale_factors])

    # Scale the points so that the best metric has s=250.
    unity = 250 * min(metrics)
    scat = ax.scatter(Lambdas, scale_factor_indices, c=metrics, s=unity/metrics,
        cmap=plt.cm.plasma, vmin=0.04, vmax=0.11, **kwargs)
    ax.set_yticks(np.arange(len(unique_scale_factors)))
    ax.set_yticklabels([r"${0:.1f}$".format(_) for _ in unique_scale_factors])
    ax.set_ylim(-1, len(unique_scale_factors))

    # Draw a circle around the best three.
    #for index, color in zip(np.argsort(metrics), ("k", )):#"#AAAAAA", "#BBBBBB", "#CCCCCC", "#DDDDDD")):
    #    ax.scatter([Lambdas[index]], [scale_factor_indices[index]],
    #        s=450, edgecolor=color, facecolor="w", zorder=-1, linewidths=2)
        

    ax.semilogx()

    for _ in np.arange(len(unique_scale_factors) - 1):
        ax.axhline(_ + 0.5, c="#EEEEEE", zorder=-1)

    cbar = plt.colorbar(scat)
    cbar.set_label(metric_label or r"Metric")
    cbar.set_ticks([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11])

    ax.set_xlabel(r"$\rm{Regularization},$ $\Lambda$")
    ax.set_ylabel(r"$\rm{Scale}$ $\rm{factor},$ $f$")
    ax.yaxis.set_tick_params(width=0)

    fig.tight_layout()
    return fig

if __name__ == "__main__":

    snapshot_filenames = glob("../*.individual_visits")

    metric_labels = [
        r"$\rm{median}\{|T_{\rm eff,combined} - T_{\rm eff,individual}|\}$",
        r"$\rm{median}\{|\log{g}_{\rm combined} - \log{g}_{\rm individual}|\}$",
        r"$\rm{median}\{|[\rm{Al}/\rm{H}]_{\rm combined} - [\rm{Al}/\rm{H}]_{\rm individual}|\}$", # AL
        r"$\rm{median}\{|[\rm{Ca}/\rm{H}]_{\rm combined} - [\rm{Ca}/\rm{H}]_{\rm individual}|\}$", #'CA',
        r"$\rm{median}\{|[\rm{C}/\rm{H}]_{\rm combined} - [\rm{C}/\rm{H}]_{\rm individual}|\}$", #'C',
        r"$\rm{median}\{|[\rm{Fe}/\rm{H}]_{\rm combined} - [\rm{Fe}/\rm{H}]_{\rm individual}|\}$", #'FE',
        r"$\rm{median}\{|[\rm{K}/\rm{H}]_{\rm combined} - [\rm{K}/\rm{H}]_{\rm individual}|\}$", #'K',
        r"$\rm{median}\{|[\rm{Mg}/\rm{H}]_{\rm combined} - [\rm{Mg}/\rm{H}]_{\rm individual}|\}$", #'MG',
        r"$\rm{median}\{|[\rm{Mn}/\rm{H}]_{\rm combined} - [\rm{Mn}/\rm{H}]_{\rm individual}|\}$", #'MN',
        r"$\rm{median}\{|[\rm{Na}/\rm{H}]_{\rm combined} - [\rm{Na}/\rm{H}]_{\rm individual}|\}$", #'NA',
        r"$\rm{median}\{|[\rm{Ni}/\rm{H}]_{\rm combined} - [\rm{Ni}/\rm{H}]_{\rm individual}|\}$", #'NI',
        r"$\rm{median}\{|[\rm{N}/\rm{H}]_{\rm combined} - [\rm{N}/\rm{H}]_{\rm individual}|\}$", #'N',
        r"$\rm{median}\{|[\rm{O}/\rm{H}]_{\rm combined} - [\rm{O}/\rm{H}]_{\rm individual}|\}$", #'O',
        r"$\rm{median}\{|[\rm{Si}/\rm{H}]_{\rm combined} - [\rm{Si}/\rm{H}]_{\rm individual}|\}$", #'SI',
        r"$\rm{median}\{|[\rm{S}/\rm{H}]_{\rm combined} - [\rm{S}/\rm{H}]_{\rm individual}|\}$", #'S',
        r"$\rm{median}\{|[\rm{Ti}/\rm{H}]_{\rm combined} - [\rm{Ti}/\rm{H}]_{\rm individual}|\}$", #'TI',
        r"$\rm{median}\{|[\rm{V}/\rm{H}]_{\rm combined} - [\rm{V}/\rm{H}]_{\rm individual}|\}$", #'V'
    ]
    label_names = [
        "TEFF",
        "LOGG",
        "AL",
        "CA",
        "C",
        "FE",
        "K",
        "MG",
        "MN",
        "NA",
        "NI",
        "N",
        "O",
        "SI",
        "S",
        "TI",
        "V"
    ]

    """
    figures = []
    for i, (metric_label, label_name) in enumerate(zip(metric_labels, label_names)):

        def metric(snrs, high_snr_expected, high_snr_inferred, 
            differences_expected, differences_inferred, single_visit_inferred):
            return np.sum(np.abs(differences_expected[:, i]))

        fig = plot_test_scalar_metrics(metric, snapshot_filenames, metric_label)
        
        fig.savefig("gs-mad-{0}.png".format(label_name), dpi=300)
    """
    def metric(snrs, high_snr_expected, high_snr_inferred, 
        differences_expected, differences_inferred, single_visit_inferred):
        return np.median(np.abs(differences_expected[:, 2:]))

    metric_label = r"${\rm median}\left(\left|[\rm{X}/\rm{H}]_{\rm combined} - [\rm{X}/\rm{H}]_{\rm individual}\right|\right)$"
    fig = plot_test_scalar_metrics(metric, snapshot_filenames, metric_label)
    fig.savefig("gs-mad-all-elements.pdf", dpi=300)

    raise a