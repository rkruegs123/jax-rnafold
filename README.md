# JAX-RNAfold

This repository contains relevant code for the paper under submission titled "Differentiable Partition Function Calculation for RNA." In this work, we introduce an algorithm for computing the partition function of an RNA sequence that is differentiable. While we demonstrate the utility of this method for designing sequences with a desired secondary structure, our method can be generalized to any loss function of the partition function. Additionally, our current implementation has a high memory cost and therefore can only be run for short/medium length sequences (<50 nt). Stay tuned for an optimized implementation!

To reproduce the results of our method on the Eterna100 structures of length at most 50, please create a `data/` directory in the base directory and run `python3 src/design.py` from the base directory. Be sure to install JAX per the [installation guide](https://github.com/google/jax#installation).

This code is made available for research purposes only and is not to be used for commercial purposes of any kind.

## Citation

Please use refer to the following DOI's for appropriately citing this work:
- This codebase: [![DOI](https://zenodo.org/badge/593600922.svg)](https://zenodo.org/badge/latestdoi/593600922)
- Our bioRxiv paper: [![DOI:10.1101/2023.01.30.526001](http://img.shields.io/badge/DOI-10.1101/2023.01.30.526001-B31B1B.svg)](https://doi.org/10.1101/2023.01.30.526001)

