Energy Model
============

.. default-role:: code

The nearest neighbor model decomposes an RNA secondary structure into various loop
motifs. The energy of a sequence/structure pair can then be computed as the sum of
the energy contributions of each loop.

JAX RNAfold provides a standard format for defining such an energy model via the
`Model` base class. It provides a template for writing energy models by defining
abstract methods for each energy contribution. In this way, methods that compute
thermodynamic quantities (such as those for computing the partition function, or
for computing the energy of a given sequence/structure pair) can simply take a
`Model` class as input and therefore be implemented independently of the specific
details of a given energy model.

This allows us to define a set of different energy models, which is particularly
helpful for testing. For example, the `All1Model` can be used to check that
the partition function is enumerating the correct number of loop motifs, and
the `RandomModel` can be used to expose bugs in aspects of the energy model
that would otherwise contribute in a negligible manner. Finally, the true
nearest neighbor model with no explicit treatment of coaxial stacks, terminal
mismatches, or dangling ends (CTDs) is implemented in the `StandardNNModel`
and `JaxNNModel`. The latter is equivalent to the former but is compatible with
JIT-compilation in JAX. Note that this implementation of the nearest neighbor
model is equivalent to the `d0` option in ViennaRNA. We also implement the `d2`
option (see `examples/eterna_example.py` for usage). Please reach out directly
if you require alternative implementations of the energy function (e.g. equivalent
to `d1` ViennaRNA).



.. automodule:: jax_rnafold.d0.energy

.. autoclass:: Model
  :members:

.. autoclass:: All1Model
.. autoclass:: RandomModel
.. autoclass:: StandardNNModel
.. autoclass:: JaxNNModel
