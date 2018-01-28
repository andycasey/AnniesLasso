.. The Cannon documentation master file, created by
   sphinx-quickstart on Mon Feb 20 22:28:06 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Cannon
==========

The Cannon is a data-driven approach to stellar label determination. 

The seminal paper describing The Cannon is `Ness et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...808...16N>`_. The name, *The Cannon*, derives from Annie Jump-Cannon, who first arranged stellar spectra in order of temperature purely by the data, without the need for stellar models. 

This software package is released as part of `Casey et al. (2016) <http://adsabs.harvard.edu/abs/2016arXiv160303040C>`_ and builds on the original implementation of The Cannon by including a number of additional features:

 - Easily construct models with complex vectorizers (e.g., cubic polynomial models with 25 labels)
 - Analytic derivatives for blazingly-fast optimization at the training step *and* the test step
 - Built-in parallelism to run the training step and the test step in parallel 
 - *Pseudo*-continuum normalization using sums of sine and cosine functions
 - L1 regularization to discover and enforce sparsity
 - Pixel censoring masks for individual labels
 - Stratified under-sampling utilities to produce a (more) balanced training set 

The Cannon is being actively developed in a `public GitHub repository <https://github.com/andycasey/AnniesLasso>`_, where you can `open an issue <https://github.com/andycasey/AnniesLasso/issues/new>`_ if you have any problems.

User Guide
----------

.. toctree::
   :maxdepth: 3

   install
   guide
   tutorials
   api


License & Attribution
---------------------

The source code is released under the MIT license. If you make use of the code, please cite both the original Ness et al. (2015) paper and Casey et al. (2016):


.. code-block:: tex

    @ARTICLE{Ness_2015
        author = {{Ness}, M. and {Hogg}, D.~W. and {Rix}, H.-W. and {Ho}, A.~Y.~Q. and 
	    {Zasowski}, G.},
        title = "{The Cannon: A data-driven approach to Stellar Label Determination}",
        journal = {\apj},
        year = 2015,
        month = jul,
        volume = 808,
        eid = {16},
        pages = {16},
        doi = {10.1088/0004-637X/808/1/16},
    }

    @ARTICLE{Casey_2016
        author = {{Casey}, A.~R. and {Hogg}, D.~W. and {Ness}, M. and {Rix}, H.-W. and 
	    {Ho}, A.~Q. and {Gilmore}, G.},
        title = "{The Cannon 2: A data-driven model of stellar spectra for detailed chemical abundance analyses}",
        journal = {ArXiv e-prints},
        archivePrefix = "arXiv",
        eprint = {1603.03040},
        year = 2016,
        month = mar,
    }

Here is a list of notable publications that have used or developed upon The Cannon:

 - Ho et al.
 - others 
