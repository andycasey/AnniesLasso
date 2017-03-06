.. _guide:

Getting Started Guide
=====================

Before we get started, you should know that the following ingredients are required to run The Cannon: 

 - a *training set* of stars with known labels (e.g., stellar parameters and chemical abundances),
 - pseudo-continuum-normalized spectra of stars in the training set, with all spectra sampled onto the same wavelength points,
 - some *test set* spectra that you want to derive labels from, which has been processed the same way as the training set spectra.

In this guide we will provide you with the training set labels and spectra, and the test set spectra. If you want more information about `constructing a training set <tutorials.html#constructing-a-training-set>`_ or `continuum-normalizing your spectra <tutorials.html#continuum-normalization>`_, see the linked tutorials. 
 
.. note:: We use `Travis continuous integration <https://travis-ci.org/andycasey/AnniesLasso>`_ to test every change to The Cannon in Python versions 2.7, 3.5, and 3.6. The code examples here should work in any of these Python versions. 


In this guide we will train a model using `APOGEE DR12 <http://www.sdss.org/dr12/irspec/>`_ spectra and `ASPCAP <http://www.sdss.org/dr12/irspec/parameters/>`_ labels to derive effective temperature :math:`T_{\rm eff}`, surface gravity :math:`\log{g}`, and four individual chemical abundances (:math:`[{\rm Fe}/{\rm H}]`, :math:`[{\rm Na}/{\rm H}]`, :math:`[{\rm Ti}/{\rm H}]`, :math:`[{\rm Ni}/{\rm H}]`). These spectra have been pseudo-continuum-normalized using a sum of sine and cosine functions (which is a different process `ASPCAP <http://www.sdss.org/dr12/irspec/parameters/>`_ uses for normalization), and individual visits have been stacked.

Here we won't use any `regularization <tutorials.html#regularization>`_ or `wavelength censoring <tutorials.html#censoring>`_, but these can be applied at the end, and the ``CannonModel`` object can be retrained to make use of regularization and/or censoring by using the ``.train()`` function.

Downloading the data
--------------------

You can download the required data for this guide using the following command:

::

    wget zenodo-link #TODO  

Creating a model
----------------

After you have `installed The Cannon <install>`_, you can use the following Python code to construct a ``CannonModel`` object:


.. code-block:: python
    :linenos:

    from astropy.table import Table 
    from six.moves import cPickle as pickle 
    from sys import version_info

    import thecannon as tc

    # Load the training set labels.
    training_set_labels = Table.read("apogee-dr12-training-set-labels.fits")

    # Load the training set spectra.
    pkl_kwds = dict(encoding="latin-1") if version_info[0] >= 3 else {}
    with open("apogee-dr12-training-set-spectra.pkl", "rb") as fp:
        training_set_flux, training_set_ivar = pickle.load(fp, **pkl_kwds)

    # Specify the labels that we will use to construct this model.
    label_names = ("TEFF", "LOGG", "FE_H", "NA_H", "TI_H", "NI_H")
 
    # Construct a CannonModel object using a quadratic (O=2) polynomial vectorizer.
    model = tc.CannonModel(
        training_set_labels, training_set_flux, training_set_ivar,
        vectorizer=tc.vectorizer.PolynomialVectorizer(label_names, 2))


Let's check the model configuration:

.. code-block:: python

    >>> print(model)
    <tc.model.CannonModel of 6 labels with a training set of 14141 stars each with 8575 pixels>

    >>> print(model.vectorizer.human_readable_label_vector)
    1 + TEFF + LOGG + FE_H + NA_H + TI_H + NI_H + TEFF^2 + LOGG*TEFF + FE_H*TEFF + NA_H*TEFF + TEFF*TI_H + NI_H*TEFF + LOGG^2 + FE_H*LOGG + LOGG*NA_H + LOGG*TI_H + LOGG*NI_H + FE_H^2 + FE_H*NA_H + FE_H*TI_H + FE_H*NI_H + NA_H^2 + NA_H*TI_H + NA_H*NI_H + TI_H^2 + NI_H*TI_H + NI_H^2

    # This model has no regularization.
    >>> print(model.regularization)
    None

    # This model includes no censoring.
    >>> print(model.censors)
    None


The training step
-----------------

The model configuration matches what we expected, so let's train the model and make it useful:

.. code-block:: python

    >>> theta, s2, metadata = model.train(threads=1)
    2017-03-06 14:18:40,920 [INFO] Training 6-label CannonModel with 14141 stars and 8575 pixels/star
    [====================================================================================================] 100% (147s) 


This model took about two minutes to train on a single core. Pixels can be trained independently, so you can parallelize the training step to as many threads as you want using the ``threads`` keyword argument. 

The ``.train()`` function returns the :math:`\theta` coefficients, the noise residuals :math:`s^2`, and metadata associated with the training of each pixel. The :math:`\theta` coefficients and scatter terms :math:`s^2` are also accessible through the ``.theta`` and ``.s2`` attributes, respectively.

.. code-block:: python
    :linenos:

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(model.theta.T[0], c='b')
    ax.set_xlabel(r'Pixel')
    ax.set_ylabel(r'$\theta_0$')

    # Alternatively, you can use the convenient plotting functions:
    fig_theta = tc.plot.theta(model, indices=0)

    fig_s2 = tc.plot.s2(model)

# TODO --> Theta and s2 figures

The test step
-------------

The trained model can now be used to run the test step against all APOGEE spectra. First, we will run the test step *on the training set spectra* as a sanity check to ensure we can approximately recover the ASPCAP labels.

.. code-block:: python
    :linenos:

    test_labels = model.test(training_set_flux, training_set_ivar, threads=1)

    # Plot a comparison between the ASPCAP labels and the labels returned at the test step.
    fig_comparison = tc.plot.one_to_one(model, test_labels)
    

Saving the model to disk
------------------------

All ``CannonModel`` objects can be written to disk, and read from disk in order to run the test step at a later time. When a model is saved, it can either be saved with or without the training set fluxes and inverse variances. The training set fluxes and inverse variances aren't strictly needed anymore once the model is trained, but they can be useful if you want to re-train the model (e.g., with regularization or censoring), or if you want to run the test step on the spectra used to train the model. 


.. code-block:: python
   :linenos:

    model.write("apogee-dr12.model")
    model.write("apogee-dr12-full.model", include_training_set_spectra=True)


By default the training set spectra are not saved because they can add considerably to the file size. The ``apogee-dr12-complete.model`` file size would be smaller given a smaller training set.

.. code-block:: python
 
    >>> ls -lh *.model
    -rw-rw-r-- 1 arc arc 1.9G Mar  6 15:58 apogee-dr12-complete.model
    -rw-rw-r-- 1 arc arc 2.3M Mar  6 15:58 apogee-dr12.model


Any saved models can be loaded from disk using the ``.read()`` function:

.. code-block:: python

    >>> new_model = tc.CannonModel.read("apogee-dr12.model")
    >>> new_model.is_trained
    True

