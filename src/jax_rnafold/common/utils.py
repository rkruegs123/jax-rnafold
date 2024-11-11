"""Defines utility functions.

  The following utility functions are used to convert between discrete and probabilistic sequences, convert between formats for RNA structures, and define unit conversions.
"""
import pkg_resources
import pdb
import numpy as onp
import random
from collections import deque
from pathlib import Path

import jax
from jax import tree_util
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


RNA_ALPHA = "ACGU"
RNA_ALPHA_IDX = {nuc: RNA_ALPHA.index(nuc) for nuc in RNA_ALPHA}
NTS = len(RNA_ALPHA)
INVALID_BASE = len(RNA_ALPHA)+1
ALL_PAIRS = ["AU", "UA", "GC", "CG", "GU", "UG"]
NBPS = len(ALL_PAIRS)
seq_to_vec = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "U": [0, 0, 0, 1]
}
all_pairs_mat = jnp.array([[0.0, 0.0, 0.0, 1.0],  # AA, AC, AG, AU
                           [0.0, 0.0, 1.0, 0.0],  # CA, CC, CG, CU
                           [0.0, 1.0, 0.0, 1.0],  # GA, GC, GG, GU
                           [1.0, 0.0, 1.0, 0.0]]) # UA, UC, UG, UU
non_gc_pairs_mat = jnp.array([[1.0, 1.0, 1.0, 1.0],  # AA, AC, AG, AU
                              [1.0, 1.0, 0.0, 1.0],  # CA, CC, CG, CU
                              [1.0, 0.0, 1.0, 1.0],  # GA, GC, GG, GU
                              [1.0, 1.0, 1.0, 1.0]]) # UA, UC, UG, UU
au_pairs_mat = jnp.array([[0.0, 0.0, 0.0, 1.0],  # AA, AC, AG, AU
                          [0.0, 0.0, 0.0, 0.0],  # CA, CC, CG, CU
                          [0.0, 0.0, 0.0, 0.0],  # GA, GC, GG, GU
                          [1.0, 0.0, 0.0, 0.0]]) # UA, UC, UG, UU
gu_pairs_mat = jnp.array([[0.0, 0.0, 0.0, 0.0],  # AA, AC, AG, AU
                          [0.0, 0.0, 0.0, 0.0],  # CA, CC, CG, CU
                          [0.0, 0.0, 0.0, 1.0],  # GA, GC, GG, GU
                          [0.0, 0.0, 1.0, 0.0]]) # UA, UC, UG, UU

def seq_to_one_hot(seq: str) -> onp.ndarray:
    """
    Converts a discrete sequence to a probabilistic sequence.

    Args:
      seq: A discrete sequence of length `n` represented as a string from the alphabet ACGU.

    Returns:
      A one-hotted ndarray of size `[n, 4]`.
    """
    one_hot = [seq_to_vec[base] for base in seq]
    return onp.array(one_hot)


DEFAULT_HAIRPIN = 3


def get_bp_bases(bp):
    return (RNA_ALPHA.index(ALL_PAIRS[bp][0]), RNA_ALPHA.index(ALL_PAIRS[bp][1]))
bp_bases = jnp.array([get_bp_bases(i) for i in range(NBPS)])

N4 = jnp.arange(4)
N6 = jnp.arange(6)


def matching_to_db(match: list) -> str:
    """
    Converts a matching to dot-bracket notation.

    A *matching* is a representation of the RNA secondary structure of a sequence of length `n`.
    A matching `M` is represented as an array of length `n` where `M[i] = i` if the :math:`i^{th}`
    residue is unpaired, and `M[i] = j` (and `M[j] = i`) if the :math:`i^{th}` and :math:`j^{th}` residues are paired.

    Args:
      match: A matching as defined above.

    Returns:
      The dot-bracket string corresponding to the matching.
    """
    db_str = ""

    for idx, val in enumerate(match):
        if val == idx:
            db_str += "."
        elif val > idx:
            db_str += "("
        else: # val < idx
            db_str += ")"
    return db_str

def db_to_matching(db_str: str) -> int:
    """
    Converts a dot-bracket string to a matching where a matching is defined as above.

    Args:
      db_str: A secondary structure in dot-bracket notation.

    Returns:
      The matching corresponding to the dot-bracket structure.
    """

    matching = [None] * len(db_str)
    open_loops = deque()

    for i, char in enumerate(db_str):
        if char not in ['.', '(', ')']:
            raise RuntimeError(f"Invalid character: {char}")
        if char == ".":
            matching[i] = i
        elif char == "(":
            open_loops.append(i)
        else: # char == ")"
            closing = open_loops.pop()
            matching[i] = closing
            matching[closing] = i

    return matching


def valid_pair(a, b):
    if a > b:
        a, b = b, a
    if a == 'A':
        return b == 'U'
    elif a == 'C':
        return b == 'G'
    elif a == 'G':
        return b == 'C' or b == 'U'
    return False



CELL_TEMP = 310.15 # temperature that RNA is usually at
MAX_LOOP = 30
NON_GC_PAIRS = set(['AU', 'UA', 'UG', 'GU'])
VALID_PAIRS = set(['AU', 'UA', 'UG', 'GU', 'GC', 'CG'])
R = 1.98717e-3


kb = 1.98717e-3
def boltz_onp(x, t=CELL_TEMP):
    beta = 1 / (kb*t)
    return onp.exp(-beta*x)
def boltz_jnp(x, t=CELL_TEMP):
    beta = 1 / (kb*t)
    return jnp.exp(-beta*x)

def kelvin_to_celsius(t_kelvin: float) -> float:
    """
    Converts a temperature in Kelvin to Celsius.

    Args:
      t_kelvin: A temperature in Kelvin.

    Returns:
      The converted temperature in Celsius.
    """
    return t_kelvin - 273.15

def celsius_to_kelvin(t_celsius: float) -> float:
    """
    Converts a temperature in Celsius to Kelvin.

    Args:
      t_celsius: A temperature in Celsius.

    Returns:
      The converted temperature in Kelvin.
    """
    return t_celsius + 273.15

def get_rand_seq(n: int) -> str:
    """
    Samples a random discrete sequence of length `n`.

    Args:
      n: A positive integer defining the length of the desired sequence.

    Returns:
      A random discrete sequence of length `n`, represented as a string from the alphabet ACGU.
    """
    rand_seq = [random.choice(RNA_ALPHA) for _ in range(n)]
    rand_seq = "".join([str(nuc) for nuc in rand_seq])
    return rand_seq


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


MAX_PRECOMPUTE = 4000


def random_pseq(n: int) -> onp.ndarray:
    """
    Sample a random probabilistic sequence of length `n`.

    Args:
      n: A positive integer defining the length of the desired sequence.

    Returns:
      A random probabilistic sequence of length `n`, represented as an `ndarray` of size `[n, 4]`.
    """
    p_seq = onp.empty((n, 4), dtype=onp.float64)
    for i in range(n):
        p_seq[i] = onp.random.random_sample(4)
        p_seq[i] /= onp.sum(p_seq[i])
    return p_seq

def structure_tree(db):
    n = len(db)
    ch = {-1: []}
    stk = [-1]
    right = [-1]*n
    for i in range(n):
        if db[i] == '(':
            stk.append(i)
        elif db[i] == ')':
            left = stk.pop()
            if stk[-1] not in ch:
                ch[stk[-1]] = []
            ch[stk[-1]].append(left)
            right[left] = i
    return ch, right


# Parameter paths
DATA_BASEDIR = Path(pkg_resources.resource_filename("jax_rnafold", "data"))
THERMO_PARAMS_DIR = DATA_BASEDIR / "thermo-params"
TURNER_2004 = THERMO_PARAMS_DIR / "rna_turner2004.par"
TURNER_1999 = THERMO_PARAMS_DIR / "rna_turner1999.par"
ETERNA_185x = THERMO_PARAMS_DIR / "vrna185x.par"


def sample_discrete_seqs(pseq, nsamples, key):
    n = pseq.shape[0]

    def sample_indices(sample_key):
        # Generate a random number for each row
        uniform_samples = jax.random.uniform(
            sample_key, shape=(pseq.shape[0],),
            minval=0, maxval=1)

        # Compute the cumulative sum of probabilities for each row
        cumulative_probabilities = jnp.cumsum(pseq, axis=1)

        # Determine the index of the first occurrence where the cumulative probability exceeds the random sample
        sampled_indices = jnp.sum(cumulative_probabilities < uniform_samples[:, None], axis=1)

        return sampled_indices

    sample_keys = jax.random.split(key, nsamples)
    seqs = list()
    freqs = onp.zeros((n, 4))
    for i in range(nsamples):
        sample_key = sample_keys[i]
        sample = sample_indices(sample_key)
        freqs[onp.arange(n), sample] += 1
        seq = ''.join([RNA_ALPHA[idx] for idx in sample])
        seqs.append(seq)

    # Can be used for testing -- this should approximate pseq
    freqs_norm = freqs / freqs.sum(axis=1, keepdims=True)

    return seqs, freqs_norm

def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)


def get_optimizer(optimizer_type, lr):

    if optimizer_type == "rms-prop":
        optimizer = optax.rmsprop(learning_rate=lr)
    elif optimizer_type == "lamb":
        optimizer = optax.lamb(learning_rate=lr)
    elif optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr)
    elif optimizer_type == "adagrad":
        optimizer = optax.adagrad(learning_rate=lr)
    elif optimizer_type == "adamax":
        optimizer = optax.adamax(learning_rate=lr)
    elif optimizer_type == "novograd":
        optimizer = optax.novograd(learning_rate=lr)
    elif optimizer_type == "yogi":
        optimizer = optax.yogi(learning_rate=lr)
    elif optimizer_type == "noisy-sgd":
        optimizer = optax.noisy_sgd(learning_rate=lr)
    elif optimizer_type == "fromage":
        optimizer = optax.fromage(learning_rate=lr)
    else:
        raise RuntimeError(f"Invalid choice of optimizer: {optimizer_type}")
    return optimizer



def relative_change(init_val, final_val):
    return (final_val - init_val) / (init_val)
