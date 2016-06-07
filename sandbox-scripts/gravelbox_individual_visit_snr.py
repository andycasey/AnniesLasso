
"""
Analyse individual snapshot spectra from APOGEE.
"""

import numpy as np
import os
from six.moves import cPickle as pickle
from astropy.table import Table

import AnniesLasso as tc


model_filename = "apogee-rg-custom-17label-REGULARIZED-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6.pickle.backup"
validation_filename = "apogee-rg-custom-17label-REGULARIZED-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6-VALIDATED.pickle"
output_filename = "apogee-rg-custom-17label-REGULARIZED-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6-INDIVIDUAL-VISITS.pickle"
print("what")

model_filename = "apogee-rg-17label-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6.pickle.backup"
validation_filename = "apogee-rg-17label-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6-VALIDATED.pickle"
output_filename = "apogee-rg-17label-theta_linalg_init-bfgs_factr0.1_pgtol1e-6-fmin_xtol-1e-6-ftol-1e-6-INDIVIDUAL-VISITS.pickle"

#model_filename = "gridsearch-0.5-4.0.model"
#validation_filename = "gridsearch-0.5-4.0.model.validation"
#output_filename = "gridsearch-0.5-4.0.model.individual_visits"

from glob import glob
filenames = [(_, "{}.validation".format(_), "{}.individual_visits".format(_)) \
    for _ in glob("gridsearch*s2*.model")]

with open("apogee-rg-individual-visit-normalized.pickle", "rb") as fp:
    individual_visits = pickle.load(fp, encoding="latin-1")

# Data.
PATH, CATALOG, FILE_FORMAT = ("", "apogee-rg.fits",
    "apogee-rg-custom-normalization-{}.memmap")

# Load the data.
labelled_set = Table.read(os.path.join(PATH, CATALOG))
dispersion = np.memmap(os.path.join(PATH, FILE_FORMAT).format("dispersion"),
    mode="c", dtype=float)
normalized_flux = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("flux"),
    mode="c", dtype=float).reshape((len(labelled_set), -1))
normalized_ivar = np.memmap(
    os.path.join(PATH, FILE_FORMAT).format("ivar"),
    mode="c", dtype=float).reshape(normalized_flux.shape)

elements = [label_name for label_name in labelled_set.dtype.names \
    if label_name not in ("PARAM_M_H", "SRC_H") and label_name.endswith("_H")]

np.random.seed(123) # For reproducibility.

# Split up the data into ten random subsets.
q = np.random.randint(0, 10, len(labelled_set))

validate_set = (q == 0)
train_set = (~validate_set)


# Generate initialization points.
label_names = ["TEFF", "LOGG"] + elements
tsl = labelled_set[train_set]
initial_labels = np.array([
    # 1. The fiducial values (mean of all parameters in the training set).
    [np.median(tsl[label_name]) for label_name in label_names], 
    # 2. The fiducial Teff and logg, with the 25th percentile of all abundances.
    np.hstack([
        [np.median(tsl["TEFF"]), np.median(tsl["LOGG"])],
        [np.percentile(tsl[element_label], 25) for element_label in elements]
    ]),
    # 3. The fiducial Teff and logg, with the 75th percentile of all abundances.
    np.hstack([
        [np.median(tsl["TEFF"]), np.median(tsl["LOGG"])],
        [np.percentile(tsl[element_label], 75) for element_label in elements]
    ]),
    # 4. The 25th percentile in Teff and logg, with the training set mean of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 25), np.percentile(tsl["LOGG"], 25)],
        [np.median(tsl[element_label]) for element_label in elements]
    ]),
    # 5. The 25th percentile in Teff and logg, with the 25th percentile of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 25), np.percentile(tsl["LOGG"], 25)],
        [np.percentile(tsl[element_label], 25) for element_label in elements]
    ]),
    # 6. The 25th percentile in Teff and logg, with the 75th percentile of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 25), np.percentile(tsl["LOGG"], 25)],
        [np.percentile(tsl[element_label], 75) for element_label in elements]
    ]),
    # 7. The 75th percentile in Teff and logg, with the training set mean of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 75), np.percentile(tsl["LOGG"], 75)],
        [np.median(tsl[element_label]) for element_label in elements]
    ]),
    # 8. The 75th percentile in Teff and logg, with the 25th percentile of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 75), np.percentile(tsl["LOGG"], 75)],
        [np.percentile(tsl[element_label], 25) for element_label in elements]
    ]),
    # 9. The 75th percentile in Teff and logg, with the 75th percentile of all abundances.
    np.hstack([
        [np.percentile(tsl["TEFF"], 75), np.percentile(tsl["LOGG"], 75)],
        [np.percentile(tsl[element_label], 75) for element_label in elements]
    ]),
])


for model_filename, validation_filename, output_filename in filenames:

    print("Model filename {}".format(model_filename))
    print("Validation filename {}".format(validation_filename))
    print("Output filename {}".format(output_filename))

    # Load the model
    model = tc.load_model(model_filename, threads=8)
    if not model.is_trained:
        print("Not trained..")
        continue

    if os.path.exists(validation_filename) and os.path.exists(output_filename):
        print("Done here..")
        continue
    
    # Load any high S/N stuff from the validation set.
    if not os.path.exists(validation_filename):
        expected = model.get_labels_array(labelled_set[validate_set])
        inferred, cov, metadata = model.fit(normalized_flux[validate_set], normalized_ivar[validate_set], full_output=True,
            initial_labels=initial_labels)

        with open(validation_filename, "wb") as fp:
            pickle.dump((expected, inferred, cov, metadata), fp, -1)

    else:
        with open(validation_filename, "rb") as fp:
            contents = pickle.load(fp, encoding="latin-1") 

        if len(contents) == 3:
            expected, inferred, _ = contents
        else:
            expected, inferred, cov, metadata = contents

    validation_apogee_ids = labelled_set["APOGEE_ID"][validate_set]

    if os.path.exists(output_filename): continue

    # Now fit the individual visits.
    snrs = []
    apogee_ids = []
    high_snr_expected = []
    high_snr_inferred = []
    differences_inferred = []
    differences_expected = []
    single_visit_inferred = []
    for i, apogee_id in enumerate(validation_apogee_ids):

        apogee_ids.extend([apogee_id] * len(meta["SNR"]))

        flux, ivar, meta = individual_visits[apogee_id]
        N = len(meta)

        iv_inferred = model.fit(flux, ivar, initial_labels=initial_labels)

        # difference with respect to the high S/N case.
        difference_inferred = iv_inferred - inferred[i]
        difference_expected = iv_inferred - expected[i]

        # Add these results
        snrs.extend(meta["SNR"])
        apogee_ids.extend([apogee_id] * len(meta["SNR"]))
        high_snr_expected.extend(np.tile(expected[i], N).reshape((N, -1)))
        high_snr_inferred.extend(np.tile(inferred[i], N).reshape((N, -1)))
        
        single_visit_inferred.extend(iv_inferred)

        differences_inferred.extend(difference_inferred)
        differences_expected.extend(difference_expected)

        print(i, apogee_id)
        """
        if i % 10 == 0 and i > 0:
            plt.close("all")

            # Now plot all them things.
            single_visit_inferred_ = np.array(single_visit_inferred)
            snrs_, high_snr_expected_, high_snr_inferred_, differences_inferred_, differences_expected_ \
                = map(np.array, (snrs, high_snr_expected, high_snr_inferred, differences_inferred, differences_expected))

            fig, ax = plt.subplots(1, 2)
            ok = single_visit_inferred_[:,1] < 3
            ax[0].scatter(snrs_[~ok], np.abs(differences_expected_[:, 0])[~ok], facecolor="r")
            ax[0].scatter(snrs_[ok], np.abs(differences_expected_[:, 0])[ok], facecolor="k")
            ax[0].set_title("wrt ASPCAP")

            ax[1].scatter(snrs_[~ok], np.abs(differences_inferred_[:, 0])[~ok], facecolor="r")
            ax[1].scatter(snrs_[ok], np.abs(differences_inferred_[:, 0])[ok], facecolor="k")
            ax[1].set_title("wrt TC")
            
            plt.draw()
            plt.show()
        """

    # Now plot all them things.
    single_visit_inferred = np.array(single_visit_inferred)
    snrs, high_snr_expected, high_snr_inferred, differences_inferred, differences_expected \
        = map(np.array, (snrs, high_snr_expected, high_snr_inferred, differences_inferred, differences_expected))

    data = (snrs, high_snr_expected, high_snr_inferred, differences_expected, differences_inferred, single_visit_inferred)
    with open(output_filename, "wb") as fp:
        pickle.dump(data, fp, -1)
    print("Saved to {}".format(output_filename))
