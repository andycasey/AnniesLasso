"""
Plot summary statistics as a function of S/N for stars in the training set,
showing us compared to ASPCAP for whatever labels in the model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.table import Table
from six.moves import cPickle as pickle

import AnniesLasso as tc

# Load the data.
PATH, CATALOG, FILE_FORMAT = ("", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="r", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="r", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="r", dtype=float).reshape(normalized_flux.shape)

# Split up the set.
np.random.seed(123)
q = np.random.randint(0, 10, len(labelled_set)) % 10
validate_set = (q == 0)
train_set = (q > 0)


"""
# Fit the validation stuff first (which is high S/N).
model = tc.L1RegularizedCannonModel(labelled_set[train_set],
    normalized_flux[train_set], normalized_ivar[train_set])
model.regularization = 0
model._metadata["q"] = q
model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(labelled_set,
    tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2),
    scale_factor=0.5)

model_filename = "mudbox-3label.model"
if os.path.exists(model_filename):
    model.load(model_filename)
else:
    model.train(fixed_scatter=True)
    model.save(model_filename)

inferred_labels = model.fit(normalized_flux[validate_set], normalized_ivar[validate_set])
inferred_labels = np.vstack(inferred_labels).T

fig, ax = plt.subplots(3)
for i, label_name in enumerate(model.vectorizer.label_names):
    ax[i].scatter(labelled_set[label_name][validate_set], inferred_labels[:, i])


raise a


"""






# Fit individual spectra using two different models.
with open("apogee-rg-individual-visit-normalized.pickle", "rb") as fp:
    individual_visit_spectra = pickle.load(fp, encoding="latin-1")

latex_labels = {
    "TEFF": "T_{\\rm eff}",
    "LOGG": "\log{g}",
    "FE_H": "{\\rm [Fe/H]}"
}

models_to_compare = {
    #"model1": "gridsearch-2.0-3.0.model",
    "model2": "gridsearch-2.0-3.0-s2-heuristically-set.model"
}

for model_name, saved_filename in models_to_compare.items():

    scale_factor = saved_filename
    model = tc.L1RegularizedCannonModel(None, None, None)
    with open(saved_filename, "rb") as fp:
        model._theta, model._s2, model._regularization, model._metadata = pickle.load(fp)
    model.pool = None
    model.vectorizer = tc.vectorizer.NormalizedPolynomialVectorizer(
        labelled_set,
        tc.vectorizer.polynomial.terminator(["TEFF", "LOGG", "FE_H"], 2),
        scale_factor=0.5)


    # Fit the high S/N validation data.
    #high_snr_comparison = model.fit(
    #    normalized_flux[validate_set], normalized_ivar[validate_set])

    # Fit the individual visit spectra.
    individual_visit_results = {
        "SNR": []
    }
    individual_visit_results.update(
        {label: [] for label in model.vectorizer.label_names})
    individual_visit_actual_results = {label: [] for label in model.vectorizer.label_names}

    apogee_ids = []
    N_validate_set_stars = sum(validate_set)
    for i, apogee_id in enumerate(labelled_set["APOGEE_ID"][validate_set]):

        inv_visit_flux, inv_visit_ivar, metadata = individual_visit_spectra[apogee_id]

        # Infer the labels from the individual visits.
        inferred_labels = model.fit(inv_visit_flux, inv_visit_ivar)

        # Use the labelled set as the reference scale.
        match = labelled_set["APOGEE_ID"] == apogee_id
        for j, label_name in enumerate(model.vectorizer.label_names):

            individual_visit_results[label_name].extend(
                inferred_labels[:, j] - labelled_set[label_name][match][0])
            individual_visit_actual_results[label_name].extend(
                inferred_labels[:, j]
                )
        individual_visit_results["SNR"].extend(metadata["SNR"])
        apogee_ids.extend([apogee_id] * len(metadata["SNR"]))
        print(i, apogee_id)


    # Now plot the differences.
    N_axes = len(individual_visit_results) - 1
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    K = 200
    for k, (ax, label_name) in enumerate(zip(axes, model.vectorizer.label_names)):

        x = np.log10(individual_visit_results["SNR"])
        y = np.log10(np.abs(individual_visit_results[label_name]))

        ind = np.argsort(x)

        ax.plot(x, y, "k.", alpha=0.25)
        for i in np.arange(0, len(ind)-K, K/10):
            ax.plot(np.median(x[ind][i:i+K]), np.median(y[ind][i:i+K]), "ro")
        ax.set_xlabel("log10SNR")
        ax.set_ylabel(label_name)

        ax.set_xlim(1, 2.5)

        xlim = np.array(ax.get_xlim())
        ax.plot(xlim, -xlim + np.median(x+y), 'r-')
        ax.plot(xlim, -xlim + np.mean(x+y), 'b-')

        ax.plot(xlim, 0*xlim + np.median(y), 'r')
        ax.plot(xlim, 0*xlim + np.mean(y), 'b')


        ax.set_ylim((-xlim + np.median(x+y))[::-1])

    fig, ax = plt.subplots()
    ax.scatter(
        np.array(individual_visit_actual_results["TEFF"])[rind],
        np.array(individual_visit_actual_results["LOGG"])[rind],
        c=np.array(individual_visit_actual_results["FE_H"])[rind]
        )
    # astronomers are crazy
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
