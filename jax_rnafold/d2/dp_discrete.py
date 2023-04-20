import numpy as onp
import pdb
from functools import partial
import unittest

import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap
import jax
from jax.tree_util import Partial

from jax_rnafold.common import read_vienna_params
from jax_rnafold.common.utils import HAIRPIN, MAX_LOOP
from jax_rnafold.common.utils import bcolors, CELL_TEMP, R, NON_GC_PAIRS, VALID_PAIRS
from jax_rnafold.common.utils import boltz_onp, get_rand_seq
from jax_rnafold.common import vienna_rna

from jax.config import config
config.update("jax_enable_x64", True)


vienna_params = read_vienna_params.read(postprocess=False) # read in the dictionary version, not the integer-indexed version
MIN_HAIRPIN_LENGTH = HAIRPIN

def _paired(i, j, paired_cache, multi_cache, seq, temp=CELL_TEMP):
    if j - i - 1 < MIN_HAIRPIN_LENGTH:
        return 0


    pair = seq[i] + seq[j]
    if pair not in VALID_PAIRS:
        return 0

    u = j - i - 1 # unpaired nucleotides

    # Hairpin
    hairpin_seq = seq[i:j+1] # includes i and j
    if u == 3 and hairpin_seq in vienna_params['triloops'].keys():
        sm = boltz_onp(vienna_params['triloops'][hairpin_seq], t=temp)
    elif u == 4 and hairpin_seq in vienna_params['tetraloops'].keys():
        sm = boltz_onp(vienna_params['tetraloops'][hairpin_seq], t=temp)
    elif u == 6 and hairpin_seq in vienna_params['hexaloops'].keys():
        sm = boltz_onp(vienna_params['hexaloops'][hairpin_seq], t=temp)
    else:
        hairpin_init_dg = vienna_params['hairpin'][u]
        hairpin_mismatch_dg = vienna_params['mismatch_hairpin'][pair][seq[i+1] + seq[j-1]]
        hairpin_non_gc_closing_penalty = jnp.where(pair in NON_GC_PAIRS,
                                                   vienna_params['non_gc_closing_penalty'],
                                                   0.0) # FIXME: do we need this?
        hairpin_dg = jnp.where(u == 3,
                               hairpin_init_dg + hairpin_non_gc_closing_penalty,
                               hairpin_init_dg + hairpin_mismatch_dg)
        hairpin_z = boltz_onp(hairpin_dg, t=temp)
        sm = hairpin_z

    # One branch
    ## Special case: stacking
    k = i+1
    l = j-1
    if seq[k] + seq[l] in VALID_PAIRS:
        stacking_dg = vienna_params['stack'][pair][seq[l] + seq[k]]
        stacking_z = boltz_onp(stacking_dg, t=temp)
        sm += stacking_z * paired_cache[k, l]


    ## Special case: bulge loops

    ### Right bulge, n=1
    k = i+1
    l = j-2
    if seq[k] + seq[l] in VALID_PAIRS:
        right_bulge_dg = vienna_params['bulge'][1] + vienna_params['stack'][pair][seq[l] + seq[k]]
        right_bulge_z = boltz_onp(right_bulge_dg, t=temp)
        sm += right_bulge_z * paired_cache[k, l]


    ### Right bulge, n>1
    # FIXME, range
    # MIN_HAIRPIN_LENGTH: is this the appropriate minimum distance from i+1 to the nucleotide in will base pair with?
    # j - 2: because (i+1, j-1) (and (i, j)) is stacking, and we already handled j-2 as the special case where n=1
    for l in range(k+1+MIN_HAIRPIN_LENGTH, j-2): # max(l) = j - 3
        if seq[k] + seq[l] in VALID_PAIRS:
            u = (k-i-1) + (j-l-1) # FIXME: simplify to (j-l-1). When k=i+1, (k-i-1) = (i+1-i-1) = 0
            bulge_dg = vienna_params['bulge'][u]
            if pair in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']
            if seq[l] + seq[k] in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']
            bulge_z = boltz_onp(bulge_dg, t=temp)
            sm += bulge_z * paired_cache[k, l]

    ### Left bulge, n=1
    k = i + 2
    l = j - 1
    if seq[k] + seq[l] in VALID_PAIRS:
        left_bulge_dg = vienna_params['bulge'][1] + vienna_params['stack'][pair][seq[l] + seq[k]]
        left_bulge_z = boltz_onp(left_bulge_dg, t=temp)
        sm += left_bulge_z * paired_cache[k, l]

    ### Left bulge, n>1
    for k in range(i+3, l-MIN_HAIRPIN_LENGTH):
        if seq[k] + seq[l] in VALID_PAIRS:
            u = (k-i-1) + (j-l-1) # FIXME: simplify to (j-l-1). When k=i+1, (k-i-1) = (i+1-i-1) = 0
            bulge_dg = vienna_params['bulge'][u]
            if pair in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']
            if seq[l] + seq[k] in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']
            bulge_z = boltz_onp(bulge_dg, t=temp)
            sm += bulge_z * paired_cache[k, l]


    ## Internal Loops

    ### 1x1
    k = i + 2
    l = j - 2
    if seq[k] + seq[l] in VALID_PAIRS:
        int11_dg = vienna_params['int11'][pair][seq[l] + seq[k]][seq[i+1]][seq[j-1]]
        int11_z = boltz_onp(int11_dg, t=temp)
        sm += int11_z * paired_cache[k, l]


    ### 2x1
    k = i + 3
    l = j - 2
    if seq[k] + seq[l] in VALID_PAIRS:
        int21_dg = vienna_params['int21'][seq[l] + seq[k]][pair][seq[l+1] + seq[i+1]][seq[k-1]]
        int21_z = boltz_onp(int21_dg, t=temp)
        sm += int21_z * paired_cache[k, l]


    ### 1x2. Note how we just treat this as an upside-down 2x1!
    k = i + 2
    l = j - 3
    if seq[k] + seq[l] in VALID_PAIRS:
        int12_dg = vienna_params['int21'][pair][seq[l] + seq[k]][seq[i+1] + seq[l+1]][seq[j-1]]
        int12_z = boltz_onp(int12_dg, t=temp)
        sm += int12_z * paired_cache[k, l]


    ### 2x2
    k = i + 3
    l = j - 3
    if seq[k] + seq[l] in VALID_PAIRS:
        left_nucs = seq[i+1] + seq[i+2]
        right_nucs = seq[j-2] + seq[j-1]
        int22_dg = vienna_params['int22'][pair][seq[l] + seq[k]][left_nucs][right_nucs]
        int22_z = boltz_onp(int22_dg, t=temp)
        sm += int22_z * paired_cache[k, l]


    ### 3x2 (note: same logic as 2x3)
    k = i + 4
    l = j - 3
    if seq[k] + seq[l] in VALID_PAIRS:
        n1 = k-i-1
        n2 = j-l-1
        u = n1 + n2 # FIXME: could hardcode
        initiation_dg = vienna_params['interior'][u]

        n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
        asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
        asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

        mismatch_dg = vienna_params['mismatch_interior_23'][pair][seq[i+1] + seq[j-1]]
        mismatch_dg += vienna_params['mismatch_interior_23'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

        int32_dg = initiation_dg + asymmetry_dg + mismatch_dg
        int32_z = boltz_onp(int32_dg, t=temp)
        sm += int32_z * paired_cache[k, l]

    ### 2x3
    k = i + 3
    l = j - 4
    if seq[k] + seq[l] in VALID_PAIRS:
        n1 = k-i-1
        n2 = j-l-1
        u = n1 + n2 # FIXME: could hardcode
        initiation_dg = vienna_params['interior'][u]

        n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
        asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
        asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

        mismatch_dg = vienna_params['mismatch_interior_23'][pair][seq[i+1] + seq[j-1]]
        mismatch_dg += vienna_params['mismatch_interior_23'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

        int23_dg = initiation_dg + asymmetry_dg + mismatch_dg
        int23_z = boltz_onp(int23_dg, t=temp)
        sm += int23_z * paired_cache[k, l]


    ### 1xN (for N > 2)
    # FIXME: make sure you're not doing 2x1s
    k = i + 2
    for l in range(k+MIN_HAIRPIN_LENGTH+1, j-3): # max(l) = j - 4, so min(n2) = 3
        if seq[k] + seq[l] in VALID_PAIRS:
            n1 = k-i-1
            n2 = j-l-1
            u = n1 + n2
            initiation_dg = vienna_params['interior'][u]

            n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
            asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
            asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

            mismatch_dg = vienna_params['mismatch_interior_1n'][pair][seq[i+1] + seq[j-1]]
            mismatch_dg += vienna_params['mismatch_interior_1n'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

            int1n_dg = initiation_dg + asymmetry_dg + mismatch_dg
            int1n_z = boltz_onp(int1n_dg, t=temp)
            sm += int1n_z * paired_cache[k, l]


    ### Nx1 (note: same logic as 1xN)
    l = j - 2
    for k in range(i+4, l-MIN_HAIRPIN_LENGTH): # min(k) = i + 4, so min(n1) = 3
        if seq[k] + seq[l] in VALID_PAIRS:
            n1 = k-i-1
            n2 = j-l-1
            u = n1 + n2
            initiation_dg = vienna_params['interior'][u]

            n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
            asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
            asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

            mismatch_dg = vienna_params['mismatch_interior_1n'][pair][seq[i+1] + seq[j-1]]
            mismatch_dg += vienna_params['mismatch_interior_1n'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

            intN1_dg = initiation_dg + asymmetry_dg + mismatch_dg
            intN1_z = boltz_onp(intN1_dg, t=temp)
            sm += intN1_z * paired_cache[k, l]


    ### 2xN (for n > 3)
    k = i + 3
    for l in range(k+MIN_HAIRPIN_LENGTH+1, j-4): # max(l) = j - 5, so min(n2) = 4
        if seq[k] + seq[l] in VALID_PAIRS:
            n1 = k-i-1
            n2 = j-l-1
            u = n1 + n2
            initiation_dg = vienna_params['interior'][u]

            n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
            asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
            asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

            mismatch_dg = vienna_params['mismatch_interior'][pair][seq[i+1] + seq[j-1]]
            mismatch_dg += vienna_params['mismatch_interior'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

            int2n_dg = initiation_dg + asymmetry_dg + mismatch_dg
            int2n_z = boltz_onp(int2n_dg, t=temp)
            sm += int2n_z * paired_cache[k, l]

    ### Nx2 (for n > 3)
    l = j - 3
    for k in range(i+5, l-MIN_HAIRPIN_LENGTH): # min(k) = i + 5, so min(n1) = 4
        if seq[k] + seq[l] in VALID_PAIRS:
            n1 = k-i-1
            n2 = j-l-1
            u = n1 + n2
            initiation_dg = vienna_params['interior'][u]

            n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
            asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
            asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

            mismatch_dg = vienna_params['mismatch_interior'][pair][seq[i+1] + seq[j-1]]
            mismatch_dg += vienna_params['mismatch_interior'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

            intN2_dg = initiation_dg + asymmetry_dg + mismatch_dg
            intN2_z = boltz_onp(intN2_dg, t=temp)
            sm += intN2_z * paired_cache[k, l]


    ### Normal case, but no redundancy with special cases -- NxM for N,M >= 3
    for k in range(i+4, j-4-MIN_HAIRPIN_LENGTH):
        for l in range(k+1+MIN_HAIRPIN_LENGTH, j-3):
            if seq[k] + seq[l] in VALID_PAIRS:
                n1 = k-i-1
                n2 = j-l-1
                u = n1 + n2
                initiation_dg = vienna_params['interior'][u]

                n1_n2_abs = jnp.sqrt((n1 - n2)**2) # FIXME: approximating jnp.abs
                asymmetry_dg = vienna_params['asymmetry'] * n1_n2_abs
                asymmetry_dg = jnp.min(jnp.array([vienna_params['asymmetry_max'], asymmetry_dg]))

                mismatch_dg = vienna_params['mismatch_interior'][pair][seq[i+1] + seq[j-1]]
                mismatch_dg += vienna_params['mismatch_interior'][seq[l] + seq[k]][seq[l+1] + seq[k-1]]

                int_dg = initiation_dg + asymmetry_dg + mismatch_dg
                int_z = boltz_onp(int_dg, t=temp)
                sm += int_z * paired_cache[k, l]




    # Multi-loops
    closing_pair = seq[j] + seq[i] # note we consider the *reversed* pair
    closing_pair_dg = vienna_params['ml_initiation'] + vienna_params['ml_branch']
    if closing_pair in NON_GC_PAIRS:
        closing_pair_dg += vienna_params['non_gc_closing_penalty']
    # FIXME (11.2.22) (done?): add *internal* mismatch (or dangle) -- note: will always be mismatch b/c it's internal - i+1 and j-1. Remember to access the reverse tables
    ## here, we add the *internal* mismatch -- note that it will always be a mismatch (rather than a dangle) b/c it is the internal case
    ## note that in this case, we use i+1 and j-1, and swap the 5' and 3' directions
    dangle5 = i+1
    dangle3 = j-1
    closing_pair_dg += vienna_params['mismatch_multi'][closing_pair][seq[dangle3] + seq[dangle5]]

    sm += multi_cache[i+1, j-1, 2] * boltz_onp(closing_pair_dg, t=temp) # note that the closing pair (ji in this case) is considered a branch

    return sm


def compute_multi_ij(i, j, paired_cache, multi_cache, seq, temp=CELL_TEMP):

    n = len(seq) # should really just pass around
    b = 0
    sm = multi_cache[i+1, j, b] * boltz_onp(vienna_params['ml_unpaired'], t=temp)
    # sm = multi_cache[i+1, j, b]
    for k in range(i+1, j+1): # j inclusive because we can make a branch all the way up to j
        branch_pair = seq[i] + seq[k]
        if branch_pair not in VALID_PAIRS:
            continue
        branch_dg = vienna_params['ml_branch']
        if branch_pair in NON_GC_PAIRS:
            branch_dg += vienna_params['non_gc_closing_penalty']

        # add *external* mismatch or dangle. -- i-1 and k+1. Don't reverse tables. You *can* have a dangle here (if i == 0 or k == n-1)
        dangle5 = i-1 if i > 0 else -1
        dangle3 = k+1 if k < n - 1 else -1

        if dangle5 != -1 and dangle3 != -1:
            branch_dg += vienna_params['mismatch_multi'][branch_pair][seq[dangle5] + seq[dangle3]]
        else:
            if dangle5 != -1:
                branch_dg += vienna_params['dangle5'][branch_pair][seq[dangle5]]
            if dangle3 != -1:
                branch_dg += vienna_params['dangle3'][branch_pair][seq[dangle3]]

        sm += paired_cache[i][k] * multi_cache[k+1][j][b] * boltz_onp(branch_dg, t=temp) # note that we index with b, rather than b - 1. Implicitly is max(0, b-1)
    multi_cache = multi_cache.at[i, j, b].set(sm)

    # b = 1, 2
    for b in range(1, 3):
        sm = multi_cache[i+1, j, b] * boltz_onp(vienna_params['ml_unpaired'], t=temp)
        for k in range(i+1, j+1): # note: when b=2, you will always hit a 0 for k=j
            branch_pair = seq[i] + seq[k]
            if branch_pair not in VALID_PAIRS:
                continue

            branch_dg = vienna_params['ml_branch']
            if branch_pair in NON_GC_PAIRS:
                branch_dg += vienna_params['non_gc_closing_penalty']

            # add *external* mismatch or dangle. -- i-1 and k+1. Don't reverse tables. You *can* have a dangle here (if i == 0 or k == n-1)
            dangle5 = i-1 if i > 0 else -1
            dangle3 = k+1 if k < n - 1 else -1

            if dangle5 != -1 and dangle3 != -1:
                branch_dg += vienna_params['mismatch_multi'][branch_pair][seq[dangle5] + seq[dangle3]]
            else:
                if dangle5 != -1:
                    branch_dg += vienna_params['dangle5'][branch_pair][seq[dangle5]]
                if dangle3 != -1:
                    branch_dg += vienna_params['dangle3'][branch_pair][seq[dangle3]]

            sm += paired_cache[i][k] * multi_cache[k+1][j][b-1] * boltz_onp(branch_dg, t=temp)
        multi_cache = multi_cache.at[i, j, b].set(sm)

    return multi_cache

def compute_ij_tables(i, j, paired_cache, multi_cache, seq, temp=CELL_TEMP):

    # First update the paired cache
    paired_ij = _paired(i, j, paired_cache, multi_cache, seq, temp=temp)
    paired_cache = paired_cache.at[i, j].set(paired_ij)

    # then update the multi cache with th eupdated paired cache
    multi_cache = compute_multi_ij(i, j, paired_cache, multi_cache, seq, temp=temp)

    return paired_cache, multi_cache


# Pass n explicitly for ease -- we can fix `n`, but `seq` will change
def fold(seq, n, # sequence information
         external_cache, paired_cache, multi_cache, # initial caches
         temp=CELL_TEMP
):

    for i in reversed(range(n)):
        sm = external_cache[i + 1]
        # for k in range(i+1+MIN_HAIRPIN_LENGTH, n):
        for k in range(i, n):
            # update the multi and paired caches (compute multi_cache_ij in paired)
            paired_cache, multi_cache = compute_ij_tables(i, k, paired_cache, multi_cache, seq, temp=temp)

            # Use the updated paired_cache to update the external_cache
            # Note: no branch penalty for extneral loops, but do need the AU/GU closure whenever you call the paired cache here. So, multiply this by the AU/GU closure penalty for ik

            external_pair = seq[i] + seq[k]

            if external_pair not in VALID_PAIRS:
                continue

            external_dg = 0
            # AU/UG penalty for (i, k)
            if external_pair in NON_GC_PAIRS:
                external_dg += vienna_params['non_gc_closing_penalty']

            # mismatch/dangles for *external*/*outer* direction -- i-1, k+1
            dangle5 = i-1 if i > 0 else -1
            dangle3 = k+1 if k < n - 1 else -1

            if dangle5 != -1 and dangle3 != -1:
                external_dg += vienna_params['mismatch_exterior'][external_pair][seq[dangle5] + seq[dangle3]]
            else:
                if dangle5 != -1:
                    external_dg += vienna_params['dangle5'][external_pair][seq[dangle5]]
                if dangle3 != -1:
                    external_dg += vienna_params['dangle3'][external_pair][seq[dangle3]]

            sm += paired_cache[i, k] * external_cache[k+1] * boltz_onp(external_dg, t=temp)
        external_cache = external_cache.at[i].set(sm)

    return external_cache[0]


def get_init_caches(n):
    ## External cache
    init_external_cache = onp.empty((n + 1,), dtype=onp.float64)
    init_external_cache[:] = onp.nan # for testing
    init_external_cache[n] = 1.0
    init_external_cache = jnp.array(init_external_cache, dtype=jnp.float64)

    ## Paired cache
    init_paired_cache = onp.empty((n+1, n+1), dtype=onp.float64)
    init_paired_cache[:] = onp.nan # for testing
    init_paired_cache = onp.tril(init_paired_cache, k=-1 - MIN_HAIRPIN_LENGTH).T # fill lower diagonal with 0s
    init_paired_cache = jnp.array(init_paired_cache, dtype=jnp.float64)

    ## Multi cache
    init_multi_cache = onp.empty((n+1, n+1, 3), dtype=onp.float64)
    init_multi_cache[:] = onp.nan
    il = onp.tril_indices(init_multi_cache.shape[0], k=-1 + MIN_HAIRPIN_LENGTH)
    # init_multi_cache[:, :, 0][il] = 1
    init_multi_cache[:, :, 0] = 1.0
    # init_multi_cache[:, :, 1] = onp.tril(init_multi_cache[:, :, 1], k=-MIN_HAIRPIN_LENGTH).T
    # init_multi_cache[:, :, 2] = onp.tril(init_multi_cache[:, :, 2], k=-MIN_HAIRPIN_LENGTH).T
    # init_multi_cache[:, :, 1][il] = 0.0
    # init_multi_cache[:, :, 2][il] = 0.0
    init_multi_cache[:, :, 1] = 0.0
    init_multi_cache[:, :, 2] = 0.0
    init_multi_cache[n, :, :] = 0.0
    init_multi_cache = jnp.array(init_multi_cache, dtype=jnp.float64)

    return init_external_cache, init_paired_cache, init_multi_cache

# Note: we don't `train` here because there is no notion of gradients for this form of a sequence
def compute_pf(seq, temp=CELL_TEMP):
    n = len(seq)
    # Each training step wlil use the same initialized caches, so we can precompute
    init_external_cache, init_paired_cache, init_multi_cache = get_init_caches(n)

    fold_seq = partial(fold, n=n, external_cache=init_external_cache,
                       paired_cache=init_paired_cache, multi_cache=init_multi_cache,
                       temp=temp
    )

    pf = fold_seq(seq)

    return pf


class TestDpDiscrete(unittest.TestCase):
    def fuzz_test(self, n, num_seq, tol_places):
        test_seqs = [get_rand_seq(n) for _ in range(num_seq)]

        tol = 1e-3
        for seq in test_seqs:
            print(f"Sequence: {seq}")

            pf = compute_pf(seq)
            print(f"\tComputed partition function: {pf}")

            vienna_pf = vienna_rna.get_vienna_pf(seq)
            print(f"\tVienna partition function: {vienna_pf}")

            self.assertAlmostEqual(pf, vienna_pf, places=tol_places)

    def test_vienna(self):
        n = 15
        num_seq = 10
        self.fuzz_test(n, num_seq, tol_places=12)

if __name__ == "__main__":
    unittest.main()
