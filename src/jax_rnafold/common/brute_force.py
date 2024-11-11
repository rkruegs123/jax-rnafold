"""
Brute-force algorithms for computing the partition functions of a probabilistic
sequence. These are useful for testing the :math:`\mathcal{O}(n^3)` algorithm.
"""

from typing import Callable
import numpy as onp

from jax_rnafold.common.utils import RNA_ALPHA, matching_to_db, db_to_matching, valid_pair
from jax_rnafold.common.sampling import UniformStructureSampler


def all_substruc_boltz_sum_uss(seq, energy_fn, hairpin):
    uss = UniformStructureSampler(hairpin=hairpin)
    uss.precomp(seq)
    sm = 0
    for i in range(uss.count_structures()):
        sm += energy_fn(seq, uss.get_nth(i))
    return sm

def ss_partition(p_seq: onp.ndarray, energy_fn: Callable, hairpin: int) -> float:
    """
    A method for brute-force calculating the structure-sequence partition
    function. Relies on `UniformStructureSampler` to compute all
    secondary structures for all :math:`4^n` discrete sequences.

    Args:
      p_seq: A probabilistic sequence of length `n`, represented as an `[n, 4]` ndarray.
      energy_fn: A function that computes the energy of a sequence for a given structure.
      hairpin: The minimum hairpin size.
    Returns:
      The structure-sequence partition function.
    """

    n = p_seq.shape[0]

    def seq_prob(seq):
        seq_prob = 1
        for i in range(len(seq)):
            seq_prob *= p_seq[i][RNA_ALPHA.index(seq[i])]
        return seq_prob

    def f(seq_list):
        if len(seq_list) == n:
            seq = ''.join(seq_list)
            return seq_prob(seq) * all_substruc_boltz_sum_uss(seq, energy_fn, hairpin)
        sm = 0
        for b in RNA_ALPHA:
            sm += f(seq_list + [b])
        return sm

    return f([])

def seq_partition(p_seq: onp.ndarray, struc: str, energy_fn: Callable) -> float:
    """
    A method for brute-force calculating the sequence partition function.

    Args:
      p_seq: A probabilistic sequence of length `n`, represented as an `[n, 4]` ndarray.
      struc: A secondary structure in dot-bracket notation
      energy_fn: A function that computes the energy of a sequence for a given structure.
    Returns:
      The sequence partition function.
    """

    n = p_seq.shape[0]

    match = db_to_matching(struc)

    def seq_prob(seq):
        seq_prob = 1
        for i in range(len(seq)):
            seq_prob *= p_seq[i][RNA_ALPHA.index(seq[i])]
        return seq_prob

    def f(seq_list):
        if len(seq_list) == n:
            seq = ''.join(seq_list)
            return seq_prob(seq) * energy_fn(seq, struc)
        sm = 0
        idx = len(seq_list)
        for b in RNA_ALPHA:
            if match[idx] < idx and not valid_pair(b, seq_list[match[idx]]):
                continue
            sm += f(seq_list + [b])
        return sm

    return f([])
