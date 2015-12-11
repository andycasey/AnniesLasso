
import os
import numpy as np
from astropy.table import Table

import AnniesLasso as tc

PATH = "/Users/arc/research/apogee/"
CATALOG = "apogee-rc-DR12.fits"
MEMMAP_FILE_FORMAT = "apogee-rc-{}.memmap"


N = 500
M = np.arange(0, 4.8, 0.1)
RUN_TRAINING = False
THREADS = 1

model_filename_format = "rc-sp-subset-regularized-order-{0:.1f}.pkl"

# Load up the data and use just a N subset of it.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("dispersion"),
    mode="c", dtype=float).flatten()
normalized_flux = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, MEMMAP_FILE_FORMAT).format("normalized-ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)

model = tc.RegularizedCannonModel(labelled_set, normalized_flux,
    normalized_ivar, dispersion=dispersion, threads=THREADS)

model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
    model.labelled_set, 
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "PARAM_M_H"], 2))

model_filename_format = "regularization-test-rc/model-{}.pkl"
if not os.path.exists(model_filename_format.format(0)):
        
    regularizations, chi_sq, log_det, models = model.validate_regularization(
        pixel_mask=(16812 > model.dispersion) * (model.dispersion > 16811),
        fixed_scatter=0)

    for i, model in enumerate(models):
        model.save(model_filename_format.format(i), overwrite=True)

i, models = 0, []
while os.path.exists(model_filename_format.format(i)):
    model = tc.RegularizedCannonModel(labelled_set, normalized_flux,
        normalized_ivar, dispersion=dispersion, threads=THREADS)
    model.load(model_filename_format.format(i))
    models.append(model)
    i += 1

raise a


#tc.diagnostics.pixel_regularization_effectiveness(models,
#    wavelengths=[15339.03364362,  15339.24556167,  15339.45748264,  15339.66940654,
#        15339.88133337],
#    latex_labels=[r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"])

tc.diagnostics.pixel_regularization_validation(models,
    wavelengths=[15339.03364362,  15339.24556167,  15339.45748264,  15339.66940654,
        15339.88133337])

raise a


fig = tc.diagnostics.regularization_effectiveness(model, wavelengths=[16811.5],
    latex_label_names=[r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"])
fig2 = tc.diagnostics.regularization_validation(model, wavelengths=[16811.5])
raise a

# Plot regularization effectiveness and validation for a few pixels of interest:
wavelengths = [15339.1, 16086.0716, 16811.5] # busy?, continuum, logg

for wavelength in wavelengths:
    fig = tc.diagnostics.regularization_effectiveness(
        model,
        wavelengths=[wavelength],
        latex_label_names=[r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"])
    fig.savefig(
        "regularization_effectiveness_{0:.1f}.pdf".format(wavelength), dpi=300)

    fig = tc.diagnostics.regularization_validation(
        model, wavelengths=[wavelength])
    fig.savefig(
        "regularization_validation_{0:.1f}.pdf".format(wavelength), dpi=300)

raise a




# Any pixel mask?
pixel_mask = np.ones(normalized_flux.shape[1], dtype=bool)
pixel_mask[:7750] = False
pixel_mask[7755:] = False
# pixels we want:
#15339.1, model.dispersion[4575] 16811.5,
# --> 1134, 4575, 7768

labelled_set = labelled_set[:N]
dispersion = dispersion[pixel_mask]
normalized_flux = normalized_flux[:N, pixel_mask]
normalized_ivar = normalized_ivar[:N, pixel_mask]

if RUN_TRAINING:

    # Run the regularized models.
    for m in M: # 10^0 --> 10^M
        # Describe the model.
        model = tc.RegularizedCannonModel(labelled_set, normalized_flux,
            normalized_ivar, dispersion=dispersion, threads=THREADS)
        model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
            model.labelled_set, 
            tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "PARAM_M_H"], 2))
        model.regularization = 10**m

        # Train and save.
        model.train()
        model.save(model_filename_format.format(m), overwrite=True)

# Speed testing: CannonModel, 8 threads: 58 seconds
#                RegularizedCannonModel with BFGS+no bounds, 8 threads: 120 seconds.
#                RegularizedCannonModel with BFGS+bounds, 8 threads: TOO SLOW
# ... but it is *correct*?




for i in range(5):
    fig = compare_single_pixel_theta(dispersion[i])
    fig.savefig("temp-{}.pdf".format(i), dpi=300)
    break

raise a
fig = compare_single_pixel_theta(16811.5)
fig.savefig("16811.559A_with_theta_zero_initial_guess.pdf")

raise a



"""

def theta_sensitivity(model, lines=None, normalize=True, abs_only=False):

    fontsize = 14
    N = len(model.vectorizer.terms)
    fig, axes = plt.subplots(1 + N, figsize=(8, 8), sharex=True)

    human_readable_terms = model.vectorizer.get_human_readable_label_vector(
        [r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"], mul="\cdot")

    for i, ax in enumerate(axes):
        norm = 1. if not normalize else np.max(np.abs(model.theta[:, i]))
        data = model.theta[:, i]/norm
        if abs_only: data = np.abs(data)
        ax.plot(model.dispersion, data, c='k')
        
        if normalize:        
            ax.set_ylabel(r"$\theta_{%s}/{\|\theta_{%s}\|_\max}$" % (i, i),
                fontsize=fontsize, rotation=0, labelpad=40)
            ax.yaxis.set_label_coords(-0.10, 0.05)

        else:
            if abs_only:
                ax.set_ylabel(r"$\|\theta_{%s}\|$" % i,
                    fontsize=fontsize, rotation=0, labelpad=40)
            else:
                ax.set_ylabel(r"$\theta_{%s}$" % i,
                    fontsize=fontsize, rotation=0, labelpad=40)

        ax_twin = ax.twinx()
        ax_twin.set_ylabel(r"${}$".format(human_readable_terms[i]), rotation=0,
            labelpad=40, fontsize=fontsize)
        ax_twin.set_yticks([])

        if lines is not None:
            [ax.axvline(line, c='#666666') for line in lines]

        if not ax.is_last_row():
            plt.setp(ax.get_xticklabels(), visible=False)

        if normalize:
            ax.set_ylim(-1.2, 1.2)
            ax.set_yticks([-1, +1])

        else:
            ax.yaxis.set_major_locator(MaxNLocator(3))

    ax.set_xlim(model.dispersion[0], model.dispersion[-1])

    if not normalize:
        min_ylim = min([a.get_ylim()[0] for a in axes[1:]])
        max_ylim = max([a.get_ylim()[1] for a in axes[1:]])
        for a in axes[1:]:
            a.set_ylim(min_ylim, max_ylim)

    fig.tight_layout()
    fig.subplots_adjust(left=0.14, wspace=0.05, hspace=0.05, right=0.86)

    return fig

raise a

model = tc.RegularizedCannonModel(
labelled_set[:N], normalized_flux[:N], normalized_ivar[:N],
threads=1, dispersion=dispersion)

model.load("rc-regularized-order-0.pkl")

fig = theta_sensitivity(model, lines=[15339.1, 16811.5, model.dispersion[4575]])
fig = theta_sensitivity(model, lines=[15339.1, 16811.5, model.dispersion[4575]], normalize=False,
    abs_only=True)



teff_without = compare_single_pixel_theta(15339.1)
#teff_without = compare_single_pixel_theta(15720.0)

#teff_without.savefig("teff_15339_without_median.pdf")

#logg_without = compare_single_pixel_theta(15770.0)
#logg_without.savefig("logg_15770_without_median.pdf")
#3) log g dependence ~ 15770, 16811.5 A
logg_without = compare_single_pixel_theta(16811.5, ylim=2)

#logg_without = compare_single_pixel_theta(15221.4)


continuum_without = compare_single_pixel_theta(pixel=4575)
#continuum_without.savefig("continuum_px1005_without_median.pdf")

raise a

for i in np.arange(500, 1000, 100):
    compare_single_pixel_theta(pixel=i)

raise a

def compare_full_theta():

    N = 2000
    # Compare them.
    model = tc.RegularizedCannonModel(
        labelled_set[:N], normalized_flux[:N], normalized_ivar[:N],
        threads=8)
    model.load("rc-regularized-order-0.pkl")

    N = model.theta.shape[1]
    x = np.arange(16)

    from matplotlib.ticker import MaxNLocator

    fontsize = 12

    quartiles = { j: [] for j in range(N) }
    for i in x:

        model.load("rc-regularized-order-{}.pkl".format(i))
        for j in range(N):

            data = model.theta[:, j]
            finite = np.isfinite(data)
            print(j, i, np.median(data[finite]))        
            quartiles[j].append(np.percentile(data[finite], [5, 50, 90]))


    human_readable_terms = model.vectorizer.get_human_readable_label_vector(
        [r"{T_{\rm eff}}", r"\log{g}", r"{\rm [M/H]}"], mul="\cdot")

    fig, axes = plt.subplots(N, figsize=(8, 8))
    quartiles = { k: np.array(v) for k, v in quartiles.items() }
    tx = x

    for i, ax in enumerate(axes):

        data = quartiles[i].T[1]
        lower, upper = quartiles[i].T[0], quartiles[i].T[2]

        ax.plot(tx, data, c='#2078B4', lw=2)
        ax.scatter(tx, data, facecolor='#2078B4', zorder=100, s=30)
        ax.fill_between(tx, lower, upper,
            facecolor="#A6CEE3", edgecolor='#2078B4', zorder=-1)

        ax.set_xlim(tx[0], tx[-1])
        ax.set_ylabel(r"$\theta_{{{0}}}$".format(i), fontsize=fontsize)
        ax_twin = ax.twinx()
        ax_twin.set_ylabel(r"${}$".format(human_readable_terms[i]), rotation=0,
            labelpad=40, fontsize=fontsize)
        ax_twin.set_yticks([])

        #ax.set_xscale("log", nonposx='clip')

        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\Lambda$", fontsize=fontsize)
            ax.set_xticks(tx)
            ax.set_xticklabels([r"$10^{{{:.0f}}}$".format(_) for _ in x])

        # Set the ytick labels to be at 0.25 and 0.75 the yrange.
        yrange = np.array(ax.get_ylim())
        ax.set_yticks(
            [np.mean(yrange) - np.ptp(yrange)/4., np.mean(yrange) + np.ptp(yrange)/4.])
        ax.set_yticklabels(["${0:+.2f}$".format(l) for l in ax.get_yticks()])
        
        # show it on the RHS


    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.86)
    raise a


compare_full_theta()

"""