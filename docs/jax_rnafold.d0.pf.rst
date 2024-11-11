Partition Function
==================

.. default-role:: code

JAX RNAfold offers two methods for computing the structure-sequence
partition function: (i) a reference implementation that is written
in pure Python, and (ii) a JAX implementation. Both implementations
function via an arbitrary energy model (i.e. an instance of `energy.Model`).
In the JAX implementation, a Boltzmann rescaling method is implemented
to mitigate numerical instabilities introduced by longer sequences.

.. autofunction:: jax_rnafold.d0.ss.get_ss_partition_fn
.. autofunction:: jax_rnafold.d0.reference.ss_partition
