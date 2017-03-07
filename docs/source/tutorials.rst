.. _tutorials:

Tutorials
=========

Constructing the training set
-----------------------------


Continuum normalization
-----------------------

Censoring
---------

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
