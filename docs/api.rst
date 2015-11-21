.. _api:

API
===

.. module:: AnniesLasso

Standard Cannon Model
-------------------------

This is the original Cannon implementation as described by Ness et al.

.. autoclass:: AnniesLasso.CannonModel
   :members:

Compressed-Sensing Cannon Model
-----------------------------------

For the future..

.. autoclass:: AnniesLasso.LassoCannonModel
   :members:

Abstract Cannon Model
---------------------

For convenience.

.. autoclass:: AnniesLasso.BaseCannonModel
   :inherited-members:


Utilities
---------

.. autofunction:: AnniesLasso.utils.parse_label_vector

.. autofunction:: AnniesLasso.utils.build_label_vector

.. autofunction:: AnniesLasso.utils.human_readable_label_vector

.. autofunction:: AnniesLasso.utils.is_structured_label_vector

.. autofunction:: AnniesLasso.utils.short_hash

.. autofunction:: AnniesLasso.utils.progressbar
