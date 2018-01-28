from astropy.table import Table
from six.moves import cPickle as pickle
from sys import version_info

import thecannon as tc

# Load the training set labels.
training_set_labels = Table.read("apogee-dr14-giants.fits")

# Load the training set spectra.
pkl_kwds = dict(encoding="latin-1") if version_info[0] >= 3 else {}
with open("apogee-dr14-giants-flux-ivar.pkl", "rb") as fp:
    training_set_flux, training_set_ivar = pickle.load(fp, **pkl_kwds)

# Specify the labels that we will use to construct this model.
label_names = ("TEFF", "LOGG", "FE_H", "MG_FE")#, "NA_FE", "TI_FE", "NI_FE")

# Construct a CannonModel object using a quadratic (O=2) polynomial vectorizer.
model = tc.CannonModel(
    training_set_labels, training_set_flux, training_set_ivar,
    vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2))


print(model)
#<thecannon.model.CannonModel of 6 labels with a training set of 1624 stars each with 8575 pixels>

print(model.vectorizer.human_readable_label_vector)
#1 + TEFF + LOGG + FE_H + NA_FE + TI_FE + NI_FE + TEFF^2 + LOGG*TEFF + FE_H*TEFF + NA_FE*TEFF + TEFF*TI_FE + NI_FE*TEFF + LOGG^2 + FE_H*LOGG + LOGG*NA_FE + LOGG*TI_FE + LOGG*NI_FE + FE_H^2 + FE_H*NA_FE + FE_H*TI_FE + FE_H*NI_FE + NA_FE^2 + NA_FE*TI_FE + NA_FE*NI_FE + TI_FE^2 + NI_FE*TI_FE + NI_FE^2

# This model has no regularization.
print(model.regularization)
#None

# This model includes no censoring.
print(model.censors)
#{}

model.train()

print(model.s2[300:])

print(model.theta[300:])
raise a

fig_s2 = tc.plot.scatter(model)
fig_s2.axes[0].set_xlim(0, 3500)
fig_s2.savefig("docs/source/scatter.png", dpi=300)

fig_theta = tc.plot.theta(model, 
    # Show the first 6 terms in the label vector.
    indices=range(5), xlim=(0, 3500),
    latex_label_names=[
        r"T_{\rm eff}",
        r"\log{g}",
        r"[{\rm Fe}/{\rm H}]",
        r"[{\rm Mg}/{\rm Fe}]",
#        r"[{\rm Na}/{\rm Fe}]",
#        r"[{\rm Ti}/{\rm Fe}]",
#        r"[{\rm Ni}/{\rm Fe}]",
    ])
fig_theta.savefig("docs/source/theta.png", dpi=300)


# The test step.
test_labels, cov, metadata = model.test(training_set_flux, training_set_ivar)


fig_comparison = tc.plot.one_to_one(
    model, test_labels, cov=cov,
    latex_label_names=[
        r"T_{\rm eff}",
        r"\log{g}",
        r"[{\rm Fe}/{\rm H}]",
        r"[{\rm Mg}/{\rm Fe}]",
#        r"[{\rm Na}/{\rm Fe}]",
#        r"[{\rm Ti}/{\rm Fe}]",
#        r"[{\rm Ni}/{\rm Fe}]",
    ])
fig_comparison.savefig("docs/source/one-to-one.png", dpi=300)

model.write("apogee-dr14-giants.model", 
    include_training_set_spectra=False, overwrite=True)
model.write("apogee-dr14-giants-full.model", 
    include_training_set_spectra=True, overwrite=True)

# ls -lh apogee-dr14-giants*.model
