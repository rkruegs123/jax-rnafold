import pdb
from pprint import pprint
import re
import numpy as np
from copy import deepcopy

import jax.numpy as jnp

from jax_rnafold.common.utils import NON_GC_PAIRS, RNA_ALPHA, RNA_ALPHA_IDX
from jax_rnafold.common.utils import CELL_TEMP, kb, MAX_LOOP, boltz_jnp, boltz_onp
from jax_rnafold.common.utils import TURNER_2004, TURNER_1999, ETERNA_185x
from jax_rnafold.common.utils import all_pairs_mat, MAX_PRECOMPUTE

from jax.config import config
config.update("jax_enable_x64", True)



INF = 1e6

seq_to_idx = {
    "N": -1,
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3
}
idx_to_seq = {v: k for k, v in seq_to_idx.items()}
all_pairs_sparse = jnp.where(all_pairs_mat == 1.0)
all_pairs_sparse = jnp.array(list(zip(all_pairs_sparse[0], all_pairs_sparse[1])))
all_pairs_str = [idx_to_seq[int(pair[0])] + idx_to_seq[int(pair[1])] for pair in all_pairs_sparse]


def precompute_initiation(init_table, n):
    precomputed = np.empty(n, dtype=np.float64)
    for u in range(MAX_LOOP):
        precomputed[u] = init_table[u]
    for u in range(MAX_LOOP, n):
        # extrapolated = init_table[MAX_LOOP] + np.round(1.07856 * np.log(u/MAX_LOOP), 2)
        extrapolated = init_table[MAX_LOOP] + np.floor(1.07856 * np.log(u/MAX_LOOP) * 100) / 100
        precomputed[u] = extrapolated
    # return jnp.array(precomputed)
    return precomputed


def process_stacking(stack_lines):
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN'] # ordering in .par file
    n_pairs = len(pairs)
    stack_data = {p: dict() for p in pairs}
    stack_lines = np.array([l.split() for l in stack_lines], dtype=np.float64) # split and cast to float

    # FIXME: 5'->3' directionality is unclear in the Vienna .par files. Doesn't align with NNDB as it is now.
    for i, p1 in zip(range(n_pairs), pairs): # FIXME: could just `enumerate(pairs)`
        for j, p2 in zip(range(n_pairs), pairs):
            stack_data[p1][p2] = stack_lines[i][j] / 100

    return stack_data

def process_int11(int11_lines):
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN'] # ordering in .par file
    nucs = ['N', 'A', 'C', 'G', 'U'] # ordering in .par file
    n_pairs = len(pairs)
    int11_data = dict()
    int11_lines = np.array([l.split() for l in int11_lines], dtype=np.float64) # split and cast to float
    for n_p1, p1 in enumerate(pairs):
        p1_start_idx = len(pairs) * len(nucs) * n_p1 # the first line where the first pair is p1
        int11_data[p1] = dict()
        for n_p2, p2 in enumerate(pairs):
            p1_p2_start_idx = p1_start_idx + len(nucs) * n_p2 # the first line where the first pair is p1 and the second pair is p2
            int11_data[p1][p2] = dict()
            for nuc1_idx, nuc1 in enumerate(nucs):
                p1_p2_nuc1_line_idx = p1_p2_start_idx + nuc1_idx # the line with pair1,pair2,nuc1
                p1_p2_nuc1_line = int11_lines[p1_p2_nuc1_line_idx]
                int11_data[p1][p2][nuc1] = dict()
                for nuc2_idx, nuc2 in enumerate(nucs):
                    int11_data[p1][p2][nuc1][nuc2] = p1_p2_nuc1_line[nuc2_idx] / 100
    return int11_data

def process_int21(int21_lines):
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN'] # ordering in .par file
    nucs = ['N', 'A', 'C', 'G', 'U'] # ordering in .par file
    n_pairs = len(pairs)
    int21_data = dict()
    int21_lines = np.array([l.split() for l in int21_lines], dtype=np.float64) # split and cast to float
    for n_p1, p1 in enumerate(pairs):
        p1_start_idx = len(pairs) * len(nucs)**2 * n_p1
        int21_data[p1] = dict()
        for n_p2, p2 in enumerate(pairs):
            p1_p2_start_idx = p1_start_idx + len(nucs)**2 * n_p2
            int21_data[p1][p2] = dict()
            for nuc1_idx, nuc1 in enumerate(nucs):
                for nuc2_idx, nuc2 in enumerate(nucs):
                    # To be explicit, we set the key as the pair of nucleotides (i.e. the `2` in the 2x1)
                    n1n2 = nuc1 + nuc2
                    int21_data[p1][p2][n1n2] = dict()
                    p1_p2_nuc12_line_idx = p1_p2_start_idx + nuc1_idx * len(nucs) + nuc2_idx # nuc12 is shorthand for nuc1_nuc2
                    p1_p2_nuc12_line = int21_lines[p1_p2_nuc12_line_idx]
                    for nuc3_idx, nuc3 in enumerate(nucs):
                        int21_data[p1][p2][n1n2][nuc3] = p1_p2_nuc12_line[nuc3_idx] / 100
    return int21_data

def process_int22(int22_lines):
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA'] # ordering in .par file
    nucs = ['A', 'C', 'G', 'U'] # ordering in .par file
    n_pairs = len(pairs)
    int22_data = dict()
    int22_lines = np.array([l.split() for l in int22_lines], dtype=np.float64) # split and cast to float
    for n_p1, p1 in enumerate(pairs):
        p1_start_idx = len(pairs) * len(nucs)**3 * n_p1
        int22_data[p1] = dict()
        for n_p2, p2 in enumerate(pairs):
            p1_p2_start_idx = p1_start_idx + len(nucs)**3 * n_p2
            int22_data[p1][p2] = dict()
            for nuc1_idx, nuc1 in enumerate(nucs):
                for nuc2_idx, nuc2 in enumerate(nucs):
                    # To be explicit, we set the keys as the pairs of nucleotides (i.e. the `2`s in the 2x2)
                    n1n2 = nuc1 + nuc2
                    int22_data[p1][p2][n1n2] = dict()
                    for nuc3_idx, nuc3 in enumerate(nucs):
                        p1_p2_n12_n3_line_idx = p1_p2_start_idx + nuc1_idx * len(nucs)**2 + nuc2_idx * len(nucs) + nuc3_idx
                        p1_p2_n12_n3_line = int22_lines[p1_p2_n12_n3_line_idx]
                        for nuc4_idx, nuc4 in enumerate(nucs):
                            int22_data[p1][p2][n1n2][nuc3 + nuc4] = p1_p2_n12_n3_line[nuc4_idx] / 100
    return int22_data

# process mismatches for internal loops, multiloops, and exterior loops
def process_int_multi_ext_mismatches(mismatch_lines):
    mismatch_lines = np.array([l.split() for l in mismatch_lines], dtype=np.float64)
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA'] # ordering in .par file. note: ignoring NS
    mismatch_data = {p: dict() for p in pairs}
    nucs = ['N', 'A', 'C', 'G', 'U']
    for i, pair in enumerate(pairs):
        pair_start_idx = i * len(nucs)
        for nuc1_idx, nuc1 in enumerate(nucs):
            pair_n_line = mismatch_lines[pair_start_idx + nuc1_idx]
            for nuc2_idx, nuc2 in enumerate(nucs):
                # note: we save data with pairs as keys
                mismatch_data[pair][nuc1 + nuc2] = pair_n_line[nuc2_idx] / 100
    return mismatch_data


# https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/1_88_84__epars_8h_source.html
def process_mismatch_hairpin(mismatch_hairpin_lines):
    mismatch_hairpin_lines = np.array([l.split() for l in mismatch_hairpin_lines], dtype=np.float64)
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA'] # ordering in .par file
    mismatch_hairpin_data = {p: dict() for p in pairs}
    for i, pair in enumerate(pairs):
        pair_start_idx = i*5
        pair_lines = mismatch_hairpin_lines[pair_start_idx:pair_start_idx+5]
        # loop per ordering in .par file
        for l, nuc1 in zip(pair_lines, ['N', 'A', 'C', 'G', 'U']):
            for val, nuc2 in zip(l, ['N', 'A', 'C', 'G', 'U']):
                mismatch_hairpin_data[pair][nuc1 + nuc2] = val / 100
    return mismatch_hairpin_data

def process_dangle(dangle_lines):
    dangle_lines = np.array([l.split() for l in dangle_lines], dtype=np.float64)
    pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA'] # ordering in .par file
    dangle_data = {p: dict() for p in pairs}
    for i, pair in enumerate(pairs):
        for j, nuc in enumerate(['N', 'A', 'C', 'G', 'U']):
            dangle_data[pair][nuc] = dangle_lines[i][j] / 100
    return dangle_data



def postprocess_interior_mismatch(raw_interior_mismatch_data):
    int_mismatch = np.empty((4, 4, 4, 4), dtype=np.float64)
    # int_mismatch[:] = np.nan # so that we get an error for invalid lookups
    int_mismatch[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for x, bx in enumerate(RNA_ALPHA):
            for y, by in enumerate(RNA_ALPHA):
                int_mismatch[i, j, x, y] = raw_interior_mismatch_data[p1][bx+by]
    return jnp.array(int_mismatch)

# Note: stuff has to be put in jax arrays, so we don't keep as dictionaries
# Note: use RNA_ALPHA indexing for postprocessing
def postprocess_data(data, max_precompute, temp=temp):

    # postprocessed_data = deepcopy(data)
    postprocessed_data = dict()

    # Stacking
    stacking = np.empty((4, 4, 4, 4), dtype=np.float64)
    # stacking[:] = np.nan # so that we get an error for invalid lookups
    stacking[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for p2 in all_pairs_str:
            k = RNA_ALPHA_IDX[p2[0]]
            l = RNA_ALPHA_IDX[p2[1]]
            stacking[i, j, k, l] = data['stack'][p1][p2]
    stacking = jnp.array(stacking)
    postprocessed_data['stack'] = stacking

    # Mismatch hairpin
    mismatch_hairpin = np.empty((4, 4, 4, 4), dtype=np.float64)
    # mismatch_hairpin[:] = np.nan # so that we get an error for invalid lookups
    mismatch_hairpin[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for k, bk in enumerate(RNA_ALPHA):
            for l, bl in enumerate(RNA_ALPHA):
                mismatch_hairpin[i, j, k, l] = data['mismatch_hairpin'][p1][bk + bl]
    mismatch_hairpin = jnp.array(mismatch_hairpin)
    postprocessed_data['mismatch_hairpin'] = mismatch_hairpin

    # int11
    int11 = np.empty((4, 4, 4, 4, 4, 4), dtype=np.float64)
    # int11[:] = np.nan
    int11[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for p2 in all_pairs_str:
            k = RNA_ALPHA_IDX[p2[0]]
            l = RNA_ALPHA_IDX[p2[1]]
            # Note: ip1 and jm1 refer to the indices of bip1 and bjm1, not the integers i+1 and j-1
            for ip1, bip1 in enumerate(RNA_ALPHA):
                for jm1, bjm1 in enumerate(RNA_ALPHA):
                    int11[i, j, k, l, ip1, jm1] = data['int11'][p1][p2][bip1][bjm1]
    int11 = jnp.array(int11)
    postprocessed_data['int11'] = int11

    # int21
    int21 = np.empty((4, 4, 4, 4, 4, 4, 4), dtype=np.float64)
    # int21[:] = np.nan
    int21[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for p2 in all_pairs_str:
            k = RNA_ALPHA_IDX[p2[0]]
            l = RNA_ALPHA_IDX[p2[1]]
            for x, bx in enumerate(RNA_ALPHA):
                for y, by in enumerate(RNA_ALPHA):
                    for z, bz in enumerate(RNA_ALPHA):
                        int21[i, j, k, l, x, y, z] = data['int21'][p1][p2][bx+by][bz]
    int21 = jnp.array(int21)
    postprocessed_data['int21'] = int21


    # int22
    int22 = np.empty((4, 4, 4, 4, 4, 4, 4, 4), dtype=np.float64)
    # int22[:] = np.nan
    int22[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for p2 in all_pairs_str:
            k = RNA_ALPHA_IDX[p2[0]]
            l = RNA_ALPHA_IDX[p2[1]]
            for w, bw in enumerate(RNA_ALPHA):
                for x, bx in enumerate(RNA_ALPHA):
                    for y, by in enumerate(RNA_ALPHA):
                        for z, bz in enumerate(RNA_ALPHA):
                            int22[i, j, k, l, w, x, y, z] = data['int22'][p1][p2][bw+bx][by+bz]
    int22 = jnp.array(int22)
    postprocessed_data['int22'] = int22

    # interior mismatches
    postprocessed_data['mismatch_interior'] = postprocess_interior_mismatch(data['mismatch_interior'])
    postprocessed_data['mismatch_interior_1n'] = postprocess_interior_mismatch(data['mismatch_interior_1n'])
    postprocessed_data['mismatch_interior_23'] = postprocess_interior_mismatch(data['mismatch_interior_23'])

    # multiloop mismatches
    mismatch_multi = np.empty((4, 4, 4, 4), dtype=np.float64)
    # mismatch_multi[:] = np.nan # so that we get an error for invalid lookups
    mismatch_multi[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for x, bx in enumerate(RNA_ALPHA):
            for y, by in enumerate(RNA_ALPHA):
                mismatch_multi[i, j, x, y] = data['mismatch_multi'][p1][bx+by]
    mismatch_multi = jnp.array(mismatch_multi)
    postprocessed_data['mismatch_multi'] = mismatch_multi

    # exterior mismatches
    mismatch_exterior = np.empty((4, 4, 4, 4), dtype=np.float64)
    # mismatch_exterior[:] = np.nan # so that we get an error for invalid lookups
    mismatch_exterior[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for x, bx in enumerate(RNA_ALPHA):
            for y, by in enumerate(RNA_ALPHA):
                mismatch_exterior[i, j, x, y] = data['mismatch_exterior'][p1][bx+by]
    mismatch_exterior = jnp.array(mismatch_exterior)
    postprocessed_data['mismatch_exterior'] = mismatch_exterior

    # dangle5 and dangle3
    dangle5 = np.empty((4, 4, 4), dtype=np.float64)
    # dangle5[:] = np.nan
    dangle5[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for x, bx in enumerate(RNA_ALPHA):
            dangle5[i, j, x] = data['dangle5'][p1][bx]
    dangle5 = jnp.array(dangle5)
    postprocessed_data['dangle5'] = dangle5

    dangle3 = np.empty((4, 4, 4), dtype=np.float64)
    # dangle3[:] = np.nan
    dangle3[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        for x, bx in enumerate(RNA_ALPHA):
            dangle3[i, j, x] = data['dangle3'][p1][bx]
    dangle3 = jnp.array(dangle3)
    postprocessed_data['dangle3'] = dangle3

    non_gc_closing_penalty_two_loops = np.empty((4, 4, 4, 4), dtype=np.float64)
    # non_gc_closing_penalty_two_loops[:] = np.nan
    non_gc_closing_penalty_two_loops[:] = INF
    for p1 in all_pairs_str:
        i = RNA_ALPHA_IDX[p1[0]]
        j = RNA_ALPHA_IDX[p1[1]]
        p1_penalty = data['non_gc_closing_penalty'] if p1 in NON_GC_PAIRS else 0.0
        for p2 in all_pairs_str:
            k = RNA_ALPHA_IDX[p2[0]]
            l = RNA_ALPHA_IDX[p2[1]]
            p2_penalty = data['non_gc_closing_penalty'] if p2 in NON_GC_PAIRS else 0.0
            non_gc_closing_penalty_two_loops[i, j, k, l] = p1_penalty + p2_penalty
    non_gc_closing_penalty_two_loops = jnp.array(non_gc_closing_penalty_two_loops)
    postprocessed_data['non_gc_closing_two_loop'] = non_gc_closing_penalty_two_loops

    # asymmetry
    asymmetry_matrix = np.empty((max_precompute, max_precompute), dtype=np.float64)
    # asymmetry_matrix[:] = np.nan
    asymmetry_matrix[:] = INF
    for n1 in range(max_precompute):
        for n2 in range(max_precompute):
            n1_n2_abs = np.abs(n1 - n2)
            n1_n2_dg = data['asymmetry'] * n1_n2_abs
            n1_n2_dg = np.min([n1_n2_dg, data['asymmetry_max']])
            asymmetry_matrix[n1, n2] = n1_n2_dg
    asymmetry_matrix = jnp.array(asymmetry_matrix)
    postprocessed_data['asymmetry_matrix'] = asymmetry_matrix

    # initiations
    postprocessed_data['bulge'] = jnp.array(data['bulge'])
    postprocessed_data['hairpin'] = jnp.array(data['hairpin'])
    postprocessed_data['interior'] = jnp.array(data['interior'])

    # scalars
    postprocessed_data['ml_branch'] = jnp.float64(data['ml_branch'])
    postprocessed_data['ml_unpaired'] = jnp.float64(data['ml_unpaired'])
    postprocessed_data['ml_initiation'] = jnp.float64(data['ml_initiation'])
    postprocessed_data['non_gc_closing_penalty'] = jnp.float64(data['non_gc_closing_penalty'])
    postprocessed_data['asymmetry'] = jnp.float64(data['asymmetry'])
    postprocessed_data['asymmetry_max'] = jnp.float64(data['asymmetry_max'])
    postprocessed_data['duplex_init'] = jnp.float64(data['duplex_init'])
    
    # special hairpins
    postprocessed_data['triloops'] = data['triloops']
    postprocessed_data['tetraloops'] = data['tetraloops']
    postprocessed_data['hexaloops'] = data['hexaloops']

    # print(set(data.keys()) - set(postprocessed_data.keys()))
    # print(set(postprocessed_data.keys()) - set(data.keys()))

    # boltzmann-ed scalars and tables
    keys_to_boltz = [
        'non_gc_closing_penalty', 'ml_initiation', 'ml_branch',
        'dangle5', 'dangle3', 
        'mismatch_multi', 'mismatch_hairpin', 
        'mismatch_interior', 'mismatch_interior_23', 'mismatch_interior_1n', 
        'hairpin', 'stack', 'bulge', 'interior', 
        'int11', 'int21', 'int22',
        'asymmetry_matrix'
    ]
    for k in keys_to_boltz:
        postprocessed_data[f"boltz_{k}"] = boltz_jnp(postprocessed_data[k], t=temp)

    return postprocessed_data


def read(param_path=TURNER_2004, max_precompute=MAX_PRECOMPUTE, 
         postprocess=True, log=False, temp=CELL_TEMP):
    with open(param_path, 'r') as f:
        param_lines = f.readlines()

    # read in chunks
    chunks = list()
    next_chunk = list()
    INF = str(1e6)
    DEF = str(-50)
    for l in param_lines:
        if l.isspace():
            # skip lines that are only whitespace
            continue

        if l[0] == "#":
            chunks.append(next_chunk)
            next_chunk = list()
        processed_line = re.sub("/\*.*\*/", "", l).strip() # remove comments
        if not processed_line or processed_line.isspace():
            # skip lines that only have comments
            continue

        processed_line = processed_line.replace("INF", INF) # set inf to 1e6
        processed_line = processed_line.replace("DEF", DEF) # set def to -50
        next_chunk.append(processed_line)
    chunks.append(next_chunk) # do the last one

    assert(not chunks[0]) # first one should be empty
    assert(len(chunks[1]) == 1 and chunks[1][0] == '## RNAfold parameter file v2.0')
    chunks = chunks[2:]

    # process the chunks we care about
    # store in an easy-to-manipulate form
    data = dict()
    for chunk in chunks:
        first_line = chunk[0]
        assert(first_line[0] == '#')

        chunk_name = first_line.split('#')[-1].strip()
        if chunk_name == "stack":
            chunk_data = chunk[1:]
            stack_data = process_stacking(chunk_data)
            data['stack'] = stack_data
        elif chunk_name == "bulge":
            chunk_data = chunk[1:]
            bulge_initiations = np.array(' '.join(chunk_data).split(), dtype=np.float64) # note: 0th index is for n=0
            bulge_initiations /= 100
            data['bulge'] = precompute_initiation(bulge_initiations, max_precompute)
        elif chunk_name == "hairpin":
            chunk_data = chunk[1:]
            hairpin_initiations = np.array(' '.join(chunk_data).split(), dtype=np.float64) # note: 0th index is for n=0
            hairpin_initiations /= 100
            data['hairpin'] = precompute_initiation(hairpin_initiations, max_precompute)
        elif chunk_name == "mismatch_hairpin":
            chunk_data = chunk[1:]
            mismatch_hairpin_data =  process_mismatch_hairpin(chunk_data)
            data['mismatch_hairpin'] = mismatch_hairpin_data
        elif chunk_name == "interior":
            chunk_data = chunk[1:]
            interior_initiations = np.array(' '.join(chunk_data).split(), dtype=np.float64) # note: 0th index is for n=0
            interior_initiations /= 100
            data['interior'] = precompute_initiation(interior_initiations, max_precompute)
        elif chunk_name == "NINIO":
            chunk_data = chunk[1:]
            assert(len(chunk_data) == 1)
            ninio_data = np.array(chunk_data[0].split(), dtype=np.float64)
            data['asymmetry'] = ninio_data[0] / 100
            data['asymmetry_max'] = ninio_data[2] / 100
        elif chunk_name == "Misc":
            chunk_data = chunk[1:]
            assert(len(chunk_data) == 1)
            misc_data = np.array(chunk_data[0].split(), dtype=np.float64)
            data['duplex_init'] = misc_data[0] / 100
            data['non_gc_closing_penalty'] = misc_data[2] / 100
        elif chunk_name == "ML_params":
            chunk_data = chunk[1:]
            assert(len(chunk_data) == 1)
            ml_params_data = np.array(chunk_data[0].split(), dtype=np.float64)
            data['ml_unpaired'] = ml_params_data[0] / 100
            data['ml_initiation'] = ml_params_data[2] / 100
            data['ml_branch'] = ml_params_data[4] / 100
        elif chunk_name == "int11":
            chunk_data = chunk[1:]
            int11_data = process_int11(chunk_data)
            data['int11'] = int11_data
        elif chunk_name == "int21":
            chunk_data = chunk[1:]
            int21_data = process_int21(chunk_data)
            data['int21'] = int21_data
        elif chunk_name == "int22":
            chunk_data = chunk[1:]
            int22_data = process_int22(chunk_data)
            data['int22'] = int22_data
        elif chunk_name in ["mismatch_interior", "mismatch_interior_1n", "mismatch_interior_23"]:
            chunk_data = chunk[1:]
            mismatch_int_data = process_int_multi_ext_mismatches(chunk_data)
            data[chunk_name] = mismatch_int_data
        elif chunk_name == "mismatch_multi":
            chunk_data = chunk[1:]
            mismatch_multi_data = process_int_multi_ext_mismatches(chunk_data)
            data['mismatch_multi'] = mismatch_multi_data
        elif chunk_name == "mismatch_exterior":
            chunk_data = chunk[1:]
            mismatch_ext_data = process_int_multi_ext_mismatches(chunk_data)
            data['mismatch_exterior'] = mismatch_ext_data
        elif chunk_name in ["dangle5", "dangle3"]:
            chunk_data = chunk[1:]
            dangle_data = process_dangle(chunk_data)
            data[chunk_name] = dangle_data
        elif chunk_name in ["Hexaloops", "Tetraloops", "Triloops"]:
            chunk_name = chunk_name.lower()
            chunk_data = chunk[1:]
            special_hairpin_map = dict()
            for case in chunk_data:
                seq, dg, dh = case.split()
                special_hairpin_map[seq] = float(dg) / 100
            data[chunk_name] = special_hairpin_map
        else:
            if log:
                print(f"WARNING: unprocessed chunk name: {chunk_name}")


    if postprocess:
        data = postprocess_data(data, max_precompute, temp=temp)
    return data


class NNParams:
    def __init__(self, params_path=TURNER_2004,
                 max_precompute=MAX_PRECOMPUTE,
                 postprocess=True, log=False,
                 save_sp_hairpins_jax=False,
                 temp=CELL_TEMP
    ):
        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.params = read(params_path, max_precompute=max_precompute,
                           postprocess=postprocess, log=log, temp=self.temp)

        self.special_hexaloops = list(self.params['hexaloops'].keys())
        self.special_tetraloops = list(self.params['tetraloops'].keys())
        self.special_triloops = list(self.params['triloops'].keys())
        self.special_hairpins = self.special_hexaloops + self.special_tetraloops \
                                + self.special_triloops
        self.n_special_hairpins = len(self.special_hairpins)

        special_hairpin_energies = list()

        for _id in range(self.n_special_hairpins):
            hairpin_seq = self.special_hairpins[_id]
            u = len(hairpin_seq) - 2
            if u == 3 and hairpin_seq in self.special_triloops:
                en = self.params['triloops'][hairpin_seq]
            elif u == 4 and hairpin_seq in self.special_tetraloops:
                en = self.params['tetraloops'][hairpin_seq]
            elif u == 6 and hairpin_seq in self.special_hexaloops:
                en = self.params['hexaloops'][hairpin_seq]
            else:
                raise RuntimeError(f"Could not find energy for special hairpin: {hairpin_seq}")
            special_hairpin_energies.append(en)

        if save_sp_hairpins_jax:
            self.special_hairpin_energies = jnp.array(special_hairpin_energies, dtype=jnp.float64)
        else:
            self.special_hairpin_energies = np.array(special_hairpin_energies, dtype=np.float64)

        special_hairpin_lens = [len(sp_hairpin) for sp_hairpin in self.special_hairpins] # array 1: length of each special hairpin
        if save_sp_hairpins_jax:
            self.special_hairpin_lens = jnp.array(special_hairpin_lens)
        else:
            self.special_hairpin_lens = np.array(special_hairpin_lens)

        special_hairpin_idxs = list() # array 2: all characters concatenated
        special_hairpin_start_pos = list() # array 3: start position for each in array 2
        idx = 0
        for sp_hairpin in self.special_hairpins:
            special_hairpin_start_pos.append(idx)
            for nuc in sp_hairpin:
                special_hairpin_idxs.append(RNA_ALPHA.index(nuc))
                idx += 1
        if save_sp_hairpins_jax:
            self.special_hairpin_idxs = jnp.array(special_hairpin_idxs)
            self.special_hairpin_start_pos = jnp.array(special_hairpin_start_pos) # has size (self.n_special_hairpins,)
        else:
            self.special_hairpin_idxs = np.array(special_hairpin_idxs)
            self.special_hairpin_start_pos = np.array(special_hairpin_start_pos) # has size (self.n_special_hairpins,)



if __name__ == "__main__":
    params_2004 = read(TURNER_2004)
    params_1999 = read(TURNER_1999)
    params_vrna185x = read(ETERNA_185x)
