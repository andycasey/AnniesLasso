.. _install:

Installation
============

You can install the most recent stable version of The Cannon using `PyPI <https://pypi.python.org/pypi/the-cannon>`_ or the
development version from `GitHub <http://www.github.com/andycasey/AnniesLasso>`_.


Stable Version
--------------

The easiest way to install the most recent stable version of The Cannon is by using `pip <https://pypi.python.org/pypi/pip>`_.
This will install any of the prerequisites (e.g., `numpy <https://pypi.python.org/pypi/numpy>`_, `scipy <https://pypi.python.org/pypi/scipy>`_), if you don't already have them:

::

    pip install the-cannon

.. note:: Make sure you include the ``-`` in ``the-cannon`` in the command above, otherwise you will install `Anna Ho's version of The Cannon <https://annayqho.github.io/TheCannon/>`_, which is excellent, but does not include analytic derivatives, regularization, or censoring.

Development Version
-------------------

To get the source for the latest development version, clone the `git <https://git-scm.com/>`_ repository on `GitHub <http://www.github.com/andycasey/AnniesLasso>`_:

::

    git clone https://github.com/andycasey/AnniesLasso.git
    cd AnniesLasso
    git checkout refactor # TODO
    

Then install the package by running the following command:

::

    python setup.py install


Testing
-------

To run all the unit and integration tests, install `nose <http://nose.readthedocs.org>`_ and then run:

::

    nosetests -v --cover-package=thecannon
