import pdb
import numpy as onp
import jax.numpy as jnp
import random
from collections import deque

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
all_pairs_mat = jnp.array([[0.0, 0.0, 0.0, 1.0], # AA, AC, AG, AU
                           [0.0, 0.0, 1.0, 0.0], # CA, CC, CG, CU
                           [0.0, 1.0, 0.0, 1.0], # GA, GC, GG, GU
                           [1.0, 0.0, 1.0, 0.0]]) # UA, UC, UG, UU
non_gc_pairs_mat = jnp.array([[1.0, 1.0, 1.0, 1.0], # AA, AC, AG, AU
                              [1.0, 1.0, 0.0, 1.0], # CA, CC, CG, CU
                              [1.0, 0.0, 1.0, 1.0], # GA, GC, GG, GU
                              [1.0, 1.0, 1.0, 1.0]]) # UA, UC, UG, UU
def seq_to_one_hot(seq):
    one_hot = [seq_to_vec[base] for base in seq]
    return onp.array(one_hot)


HAIRPIN = 3


special_hexaloops = ["ACAGUACU", "ACAGUGAU", "ACAGUGCU", "ACAGUGUU"]
special_tetraloops = ["CAACGG", "CCAAGG", "CCACGG", "CCCAGG", "CCGAGG",
                      "CCGCGG", "CCUAGG", "CCUCGG", "CUAAGG", "CUACGG",
                      "CUCAGG", "CUCCGG", "CUGCGG", "CUUAGG", "CUUCGG",
                      "CUUUGG"]
special_triloops = ["CAACG", "GUUAC"]
# SPECIAL_HAIRPINS = ["CAACG", "GUUAC"]  # FIXME: Not complete
SPECIAL_HAIRPINS = special_hexaloops + special_tetraloops + special_triloops
N_SPECIAL_HAIRPINS = len(SPECIAL_HAIRPINS)
# array 1: length of each special hairpin
SPECIAL_HAIRPIN_LENS = jnp.array(
    [len(sp_hairpin) for sp_hairpin in SPECIAL_HAIRPINS]) # has size (N_SPECIAL_HAIRPINS,)
SPECIAL_HAIRPIN_IDXS = list() # array 2: all characters concatenated
SPECIAL_HAIRPIN_START_POS = list() # array 3: start position for each in array 2
idx = 0
for sp_hairpin in SPECIAL_HAIRPINS:
    SPECIAL_HAIRPIN_START_POS.append(idx)
    for nuc in sp_hairpin:
        SPECIAL_HAIRPIN_IDXS.append(RNA_ALPHA.index(nuc))
        idx += 1
SPECIAL_HAIRPIN_IDXS = jnp.array(SPECIAL_HAIRPIN_IDXS)
SPECIAL_HAIRPIN_START_POS = jnp.array(SPECIAL_HAIRPIN_START_POS) # has size (N_SPECIAL_HAIRPINS,)


def get_bp_bases(bp):
    return (RNA_ALPHA.index(ALL_PAIRS[bp][0]), RNA_ALPHA.index(ALL_PAIRS[bp][1]))
bp_bases = jnp.array([get_bp_bases(i) for i in range(NBPS)])

N4 = jnp.arange(4)


def matching_to_db(match):
    strn = []
    for i in range(len(match)):
        if match[i] < i:
            strn.append(')')
        elif match[i] > i:
            strn.append('(')
        else:
            strn.append('.')
    return ''.join(strn)


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

def get_rand_seq(n):
    rand_seq = [random.choice(RNA_ALPHA) for _ in range(n)]
    rand_seq = "".join([str(nuc) for nuc in rand_seq])
    return rand_seq

def get_rand_seq_no_special(n, max_tries=100):
    for _ in range(max_tries):
        rand_seq = get_rand_seq(n)
        contains_loop = contains_special_hairpin(rand_seq)
        if not contains_loop:
            return rand_seq
    raise RuntimeError(f"Could not find sequence of length {n} without special hairpin loop")

def matching_2_dot_bracket(matching):
    db_str = ""

    for idx, val in enumerate(matching):
        if val == idx:
            db_str += "."
        elif val > idx:
            db_str += "("
        else: # val < idx
            db_str += ")"
    return db_str

def dot_bracket_2_matching(dot_bracket_str):
    matching = [None] * len(dot_bracket_str)
    open_loops = deque()

    for i, char in enumerate(dot_bracket_str):
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


MAX_PRECOMPUTE = 200


def random_pseq(n):
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



if __name__ == "__main__":
    from jax import vmap
    fn = lambda x: x
    hi = vmap(fn, 0)(bp_bases)
    pdb.set_trace()
