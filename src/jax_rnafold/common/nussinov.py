import numpy as np

from jax_rnafold.common.utils import ALL_PAIRS, RNA_ALPHA, NTS, DEFAULT_HAIRPIN, random_pseq
from jax_rnafold.common import brute_force


def make_valid_pairs_table():
    valid_pairs = np.zeros((NTS, NTS))
    for i in range(NTS):
        for j in range(NTS):
            valid_pairs[i, j] = RNA_ALPHA[i]+RNA_ALPHA[j] in ALL_PAIRS
    return valid_pairs

def en_pair_1(bi, bj):
    return 1  # bi*37+bj

def en_pair(bi, bj):
    return bi*37+bj


def ss_partition(p_seq, en_pair=en_pair, hairpin=DEFAULT_HAIRPIN):
    n = p_seq.shape[0]

    D = np.ones((n+1, n))

    valid_pairs = make_valid_pairs_table()

    for i in range(n-1, -1, -1):
        for j in range(i, n):
            sm = D[i+1, j]
            for k in range(i+hairpin+1, j+1):
                for bi in range(NTS):
                    for bk in range(NTS):
                        if not valid_pairs[bi, bk]:
                            continue
                        sm += D[i+1, k-1]*D[k+1, j]*p_seq[i, bi] * \
                            p_seq[k, bk]*en_pair(bi, bk)
            D[i, j] = sm
    return D[0, n-1]

def energy(seq, match):
    e = 1
    for i in range(len(match)):
        if match[i] > i:
            e *= en_pair(RNA_ALPHA.index(seq[i]),
                         RNA_ALPHA.index(seq[match[i]]))
    return e
