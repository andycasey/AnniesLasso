.. _api:

API
===

Common classes and utilities in The Cannon are documented here. For more details, `view the source code <https://github.com/andycasey/AnniesLasso>`_. 

CannonModel
-----------

.. autoclass:: thecannon.CannonModel
   :members:


Censoring
---------

.. autoclass:: thecannon.censoring.Censors
   :members:

.. automodule:: thecannon.censoring
   :members:


Continuum
---------

.. automodule:: thecannon.continuum
   :members:


Fitting
-------

.. automodule:: thecannon.fitting
   :members:


Utilities
---------

.. automodule:: thecannon.utils
   :members:


Vectorizer
----------

BaseVectorizer
^^^^^^^^^^^^^^

.. automodule:: thecannon.vectorizer.base
   :members: 

PolynomialVectorizer
^^^^^^^^^^^^^^^^^^^^

.. automodule:: thecannon.vectorizer.polynomial
   :members:

``tc`` command line utility
---------------------------

The Cannon code includes a command line utility called ``tc``.

This command line tool can be used to fit spectra using a pre-trained model saved to disk, and to join many results into a single table of output labels.

.. program-output:: tc --help

The ``fit`` argument requires the following input:

.. program-output:: tc fit --help

Once the test step is complete, the results from individual files will be saved to disk. For example, if a spectrum was saved to disk as ``spectrum.pkl``, then the command ``tc fit cannon.model spectrum.pkl`` would produce an output file called ``spectrum.pkl.result``. The ``tc join`` command can then collate the output from many ``*.result`` files into a single table:

.. program-output:: tc join --help
 
