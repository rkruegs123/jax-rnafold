# JAX-RNAfold

### Automatic differentiable RNA folding in JAX

This paper contains code for *differentiable RNA folding*, a recently developed method for RNA design in which a probabilistic sequence representation is optimized via gradient descent (see the original paper [here](https://academic.oup.com/nar/article/52/3/e14/7457012)). We provide a highly optimized version of the algorithm. This algorithm can be used to design an RNA sequence to minimize an arbitrary (continuous and differentiable) function of the partition function. Gradient calculation currently scales to sequences at most 1250 nucleotides in length on a single NVIDIA 80 GB A100 GPU.


### Getting Started

To install JAX RNAfold locally, install it with pip:
```
git clone https://github.com/rkruegs123/jax-rnafold
cd jax-rnafold
pip install -e .
```
Be sure to install JAX per the [installation guide](https://github.com/google/jax#installation).

We use the `unittest` library for testing. After installation, you may run all tests via `python -m unittest discover -s tests -v` from the base directory.

We provide two example scripts in `examples/` for users to get up and running. The first, `eterna_example.py`, is for designing sequences to fold into a target secondary structure from the Eterna100 dataset. The second, `mrna_example.py`, is for mRNA design.

We also provide documentation for a more complete description.
We use `sphinx` to automate documentation.
If you want to include the relevant `sphinx` dependencies when installing the package in editable mode, run
```
pip install -e ".[docs]"
```
If you have already installed the package and only want to install the relevant `sphinx` dependencies, run
```
pip install .[docs] --no-deps
```
You can then build the documentation and explore it in your local web browser as follows:
```
cd docs/
make html # creates _build/ directory
open _build/html/index.html
```



# Publications

JAX RNAfold has been used in the following publications. If you don't see your paper on the list, but you used JAX RNAfold let us know and we'll add it to the list!

1. [Differentiable partition function calculation for RNA. (NAR 2024)](https://academic.oup.com/nar/article/52/3/e14/7457012)<br> M. C. Matthies, R. Krueger, A. E. Torda, and M. Ward
2. [Scalable Differentiable Folding for mRNA Design.](https://www.biorxiv.org/content/10.1101/2024.05.29.594436v1)<br> R. Krueger and M. Ward


# Citation

If you wish to cite the original algorithm, please cite the following .bib:
```
@article{matthies2024differentiable,
  title={Differentiable partition function calculation for RNA},
  author={Matthies, Marco C and Krueger, Ryan and Torda, Andrew E and Ward, Max},
  journal={Nucleic Acids Research},
  volume={52},
  number={3},
  pages={e14--e14},
  year={2024},
  publisher={Oxford University Press}
}
```
If you wish to cite this software package and the scaled algorithm, please cite the following .bib:
```
@article{krueger2024scalable,
  title={Scalable Differentiable Folding for mRNA Design},
  author={Krueger, Ryan and Ward, Max},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
