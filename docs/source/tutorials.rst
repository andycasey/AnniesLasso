.. _tutorials:

Tutorials
=========

Constructing the training set
-----------------------------

You'll want a balanced training set. #TODO


Continuum normalization
-----------------------

*Continuum normalization* is the process of removing the black body and instrumental effects of the spectrograph, such that the flux values are normalized between 0 and 1. 

There are a lot of aspects to continuum normalization, which are beyond the scope of this (or any?) tutorial. The crucial aspects are to ensure that the normalization procedure is a linear operation (e.g., not a high-order polynomial), and that the normalization procedure is invariant with respect to both the labels and the spectrum S/N. For example, a normalization procedure that is based on smoothing nearby pixels will give different continuum normalization for low- and high-S/N spectra of the same star, and will yield a different continuum normalization for metal-rich and metal-poor stars.  If the continuum normalization procedure is dependent on the S/N of the spectra then the (high S/N) training set will have a different continuum normalization to the (lower S/N) test set.

One example of a (pseudo) continuum normalization procedure that meets these constraints is by fitting the continuum with a sum of sine and cosine functions. This requires a list of continuum pixels to be specified in advance. However in principle, these *"continuum pixels"* can be chosen at random, as long as the same pixels are used for the continuum normalization of the training set *and* the test set.

.. code-block:: python

    import thecannon as tc

    # The dispersion and flux arrays refer to the wavelength and flux values for 
    # a single spectrum, and the ivar array refers to the inverse variance of those
    # flux values.

    # Here the continuum_pixels is a boolean array the same size as the dispersion
    # array, indicating whether each entry is a continuum pixel

    # The regions parameter can be used to specify -- for example -- separate CCDs
    # Continuum will be fit separately to each region

    continuum, metadata = tc.continuum.sines_and_cosines(dispersion, flux, ivar,
        continuum_pixels, L=1400, order=3, regions=[
            [15090, 15822],
            [15823, 16451],
            [16452, 16971]
        ])


Censoring
---------

Censoring allows for labels to be prevented from contributing to the flux at specific pixels.

For example, given a four-label model (:math:`T_{\rm eff}`, :math:`\log{g}`, :math:`[{\rm Fe}/{\rm H}]`, :math:`[{\rm Al}/{\rm H}]`), where one label only contributes to a few pixels (e.g., there are few Al lines in this spectrum, so :math:`[{\rm Al}/{\rm H}]` should only contribute to a few pixels), we can use censoring masks to prevent the :math:`[{\rm Al}/{\rm H}]` label from contributing to the stellar flux *except* for around the known Al transition. 

Censoring masks can be provided for any label. By default, ``CannonModel`` objects have no censoring masks. Models can be trained with or without censoring, and can be re-trained after a change in the censoring masks.

.. code-block:: python

    import thecannon as tc

    model = tc.CannonModel.read("apogee-dr12.model")

    uncensored_theta = model.theta

    print(model.censors)
    >>> None

    print(model.is_trained)
    >>> True

    # Censor [Al/H] everywhere except around the Al line at X Angstroms
    # todo
    model.censors["AL_H"] = tc.censoring.create_mask(
        model.dispersion, [
            [ ],
            [ ],
        ])

    # Re-train the model.
    censored_theta, censored_s2, censored_metadata = model.train()     


Regularization
--------------

Regularization is useful for discovering and enforcing model sparsity. Without any regularization, at a given pixel there could be contributions from many (e.g., 15) different elemental abundance labels. However we might believe there is only a single, unblended atomic transition that contributes to that pixel, so all 15 elemental abundance labels cannot be contributing. Regularization (specifically we use L1 regularization) encourages the spectral derivatives :math:`\theta_{1..K}` to take on zero values. 

Formally, at the training step we fix the :math:`K`-lists of labels for the :math:`n` training set stars. At each wavelength pixel :math:`j`, we then find the parameters :math:`\boldsymbol{\theta}_j` and :math:`s_j^2` by optimizing the penalized likelihood function

.. math::

    \newcommand{\argmin}[1]{\underset{#1}{\operatorname{argmin}}\,}
    \boldsymbol{\theta}_j,s^2_j \leftarrow \argmin{\boldsymbol{\theta},s}\left[
    \sum_{n=0}^{N-1} \frac{[y_{jn}-\boldsymbol{v}(\ell_n)\cdot\boldsymbol{\theta}]^2}{\sigma^2_{jn}+s^2}
    + \sum_{n=0}^{N-1} \ln(\sigma^2_{jn}+s^2) + \Lambda{}\,Q(\boldsymbol{\theta})
    \right]

where

.. math::

    Q(\boldsymbol{\theta}) = \sum_{d=1}^{D-1} |{\theta_d}| \quad .

.. note::
    The L1 regularization does not act on the first entry in :math:`\boldsymbol{\theta}`, :math:`\theta_0`, because this is a mean flux value that we do not expect to decrease with increasing regularization. 


By default, a ``CannonModel`` object will have no regularization (:math:`\Lambda = 0`). You can set the regularization strength :math:`\Lambda` using the ``.regularization`` attribute:

.. code-block:: python

    import thecannon as tc

    model = tc.CannonModel.read("apogee-dr12.model")
   
    model.regularization = 1000.0


When using the ``.regularization`` attribute, you can either specify ``None`` (equivalent to using ``model.regularization = 0``), a float-like value, or an array of size `num_pixels` with different regularization strengths for each pixel.

Often it is convenient to train a model without any regularization, and then re-train it using a higher regularization strength. This is useful because the optimization can take longer at higher regularization strengths (particularly if strict convergence tolerances are required), and the previously solved :math:`\boldsymbol{\theta}` coefficients can be used as an initial starting guess.

.. code-block:: python

    # Without any regularization:
    model.regularization = 0.0
    nr_theta, nr_s2, nr_metadata = model.train()
 
    # Let's set a strong regularization value:
    model.regularization = 10e5
    sr_theta, sr_s2, sr_metadata = model.train()
   
    # Compare the spectral coefficients for, say, [Ti/H] in nr_theta and sr_theta:
    #TODO

Sometimes you might find that the likelihood landscape is extremely flat with strong regularization in a high dimensional label space. The flat likelihood landscape makes it difficult to optimize, and you might have (valid) concerns that the optimizer has not converged to the best coefficients possible. Be aware that the training step is a **convex** optimization problem (when :math:`s_j^2` is fixed), so the optimizer is working towards the global minimum (and not a *local minimum*), but the line search in high dimensions may find that the landscape is too flat to continue optimizing.

In these circumstances, you can switch from using the L-BFGS-B optimization algorithm (default; ``scipy.optimize.fmin_l_bfgs_b``) to using Powell's method (``scipy.optimize.fmin_powell``). The result from the previous trained model will be used as an initial guess, so the coefficients will not have far to optimize. Powell's method can have very strict tolerance requirements, and should perform well even if the likelihood landscape is very flat. However, Powell's method does not make use of analytic derivatives, so the training time will be considerably longer. 

.. code-block:: python

    # Re-train the model using Powell's method, and very strict convergence requirements.
    # (The $\theta$ values that were trained using BFGS will be used as a starting point)
    powell_theta, powell_s2, powell_metadata = model.train(
        op_method="powell", 
        op_kwds=dict(xtol=1e-8, ftol=1e-8, maxiter=100000, maxfun=100000))

If you are worried that you could have optimization convergence problems, it is a useful sanity check to perform this dumb, slow optimization: train first with BFGS (default), then re-train using Powell's method and very strict convergence criteria. The training time will take longer, but you can plot the spectral derivatives from both training steps and compare the differences!
