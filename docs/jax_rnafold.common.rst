Utilities
=========

.. default-role:: code

Below we enumerate several classes of utility functions used for the implementation
and testing of the differentiable partition function algorithm.

General Utilities
-----------------
.. automodule:: jax_rnafold.common.utils
.. autofunction:: get_rand_seq
.. autofunction:: seq_to_one_hot
.. autofunction:: random_pseq
.. autofunction:: matching_to_db
.. autofunction:: db_to_matching
.. autofunction:: kelvin_to_celsius
.. autofunction:: celsius_to_kelvin


Structure Sampling
------------------
.. automodule:: jax_rnafold.common.sampling

.. autoclass:: UniformStructureSampler
  :members:


Brute Force Calculators
-----------------------
.. automodule:: jax_rnafold.common.brute_force
.. autofunction:: ss_partition
.. autofunction:: seq_partition


Parameter Loading
-----------------
.. automodule:: jax_rnafold.common.read_vienna_params

.. autofunction:: read
.. autoclass:: NNParams
  :members:

Protein
-------

.. automodule:: jax_rnafold.common.protein

.. autoclass:: CodonFrequencyTable
  :members:
.. autofunction:: get_rand_aa_seq
.. autofunction:: random_cds
.. autofunction:: get_expected_cai_fn
