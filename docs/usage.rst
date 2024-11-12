Usage
=====

.. _installation:

Installation
------------

The following installation instructions were tested on Ubuntu 24.04 LTS with Python 3.10.

To begin, create a fresh environment and activate it. You may do so with conda, e.g.

.. code-block:: console

   conda create -n ENV-NAME python=3.10
   conda activate ENV-NAME

or with ``venv``, e.g.

.. code-block:: console

   python3 -m venv path/to/ENV-NAME
   source path/to/ENV-NAME/bin/activate

You may then install JAX RNAfold locally via pip:

.. code-block:: console

   git clone https://github.com/rkruegs123/jax-rnafold
   cd jax-rnafold
   pip install -e .

By default, the CPU version of JAX is installed but be sure to install JAX according to your available hardware accelerators per the `installation guide <https://github.com/google/jax#installation>`_.


.. _testing:

Testing
-------

We use the ``unittest`` library for testing. After installation, you may run all tests via the following command from the base directory:

.. code-block:: console

   python -m unittest discover -s tests -v

.. _examples:

Examples
--------

We provide two example scripts in ``examples/`` for users to get up and running. The first, ``eterna_example.py``, is for designing sequences to fold into a target secondary structure from the Eterna100 dataset. The second, ``mrna_example.py``, is for mRNA design.


.. _ctds:

Note on CTDs
------------

There are several choices for the treatment of coaxial stacks, terminal mismatches, and dangling ends (CTDs) in the nearest neighbor model. We implement two such choices, corresponding to the ``-d0`` and ``-d2`` options in ViennaRNA. For simplicity, we only document the ``-d0`` model as there is high similarity with ``-d2``. However, those interested in getting started with ``-d2`` can reference ``eterna_example.py`` which uses this model for structure design. The mRNA design example in ``mrna_example.py`` uses ``-d0``. Note that ``-d2`` is significantly more costly (in both memory and time) than ``-d0`` since a different algorithm must be used to handle terminal mismatches with continuous sequences, and is therefore more limited in sequence length.
