import numpy as onp
import unittest
import pdb

import jax.numpy as jnp

from jax_rnafold.common.utils import RNA_ALPHA, INVALID_BASE, ALL_PAIRS, NON_GC_PAIRS
from jax_rnafold.common.utils import SPECIAL_HAIRPINS, N_SPECIAL_HAIRPINS
from jax_rnafold.common.utils import structure_tree
from jax_rnafold.common.energy_hash import float_hash
from jax_rnafold.common.utils import boltz_onp, boltz_jnp
from jax_rnafold.common.utils import non_gc_pairs_mat, all_pairs_mat
from jax_rnafold.common.utils import kb, CELL_TEMP
from jax_rnafold.common import read_vienna_params

vienna_params = read_vienna_params.read(postprocess=False)
jax_vienna_params = read_vienna_params.read(postprocess=True)

# construct array of special hairpin energies
special_hairpin_energies = list()
for id in range(N_SPECIAL_HAIRPINS):
    hairpin_seq = SPECIAL_HAIRPINS[id]
    u = len(hairpin_seq) - 2
    if u == 3 and hairpin_seq in vienna_params['triloops'].keys():
        en =  jax_vienna_params['triloops'][hairpin_seq]
    elif u == 4 and hairpin_seq in vienna_params['tetraloops'].keys():
        en = jax_vienna_params['tetraloops'][hairpin_seq]
    elif u == 6 and hairpin_seq in vienna_params['hexaloops'].keys():
        en = jax_vienna_params['hexaloops'][hairpin_seq]
    else:
        raise RuntimeError(f"Could not find energy for special hairpin: {hairpin_seq}")
    special_hairpin_energies.append(en)
special_hairpin_energies_onp = onp.array(special_hairpin_energies, dtype=onp.float64)
special_hairpin_energies_jnp = jnp.array(special_hairpin_energies, dtype=jnp.float64)


class Model:
    def en_ext_branch(self, bi, bj):
        pass

    def en_multi_branch(self, bi, bk):
        pass

    def en_5dangle(self, bim1, bi, bj):
        pass

    def en_5dangle_inner(self, bi, bjm1, bj):
        pass

    def en_3dangle(self, bi, bj, bjp1):
        pass

    def en_3dangle_inner(self, bi, bip1, bj):
        pass

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        pass

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        pass

    def en_multi_closing(self, bi, bj):
        pass

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        pass

    def en_hairpin_special(self, id):
        # id is the index into SPECIAL_HAIRPINS
        pass

    def en_stack(self, bi, bj, bk, bl):
        pass

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        pass

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        pass

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        pass

    def en_internal_init(self, sz):
        pass

    def en_internal_asym(self, lup, rup):
        pass

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        en = self.en_internal_init(lup+rup)*self.en_internal_asym(lup, rup)*self.en_il_inner_mismatch(
            bi, bj, bip1, bjm1)*self.en_il_outer_mismatch(bk, bl, bkm1, blp1)
        return en


class All0Model(Model):
    def en_ext_branch(self, bi, bj):
        return 0

    def en_multi_branch(self, bi, bk):
        return 0

    def en_5dangle(self, bim1, bi, bj):
        return 0

    def en_5dangle_inner(self, bi, bjm1, bj):
        return 0

    def en_3dangle(self, bi, bj, bjp1):
        return 0

    def en_3dangle_inner(self, bi, bip1, bj):
        return 0

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        return 0

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        return 0

    def en_multi_closing(self, bi, bj):
        return 0

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 0

    def en_hairpin_special(self, id):
        return 0

    def en_stack(self, bi, bj, bk, bl):
        return 0

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return 0

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 0

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 0

    def en_internal_init(self, sz):
        return 0

    def en_internal_asym(self, lup, rup):
        return 0


class All1Model(Model):
    def en_ext_branch(self, bi, bj):
        return 1

    def en_multi_branch(self, bi, bk):
        return 1

    def en_5dangle(self, bim1, bi, bj):
        return 1

    def en_5dangle_inner(self, bi, bjm1, bj):
        return 1

    def en_3dangle(self, bi, bj, bjp1):
        return 1

    def en_3dangle_inner(self, bi, bip1, bj):
        return 1

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        return 1

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        return 1

    def en_multi_closing(self, bi, bj):
        return 1

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 1

    def en_hairpin_special(self, id):
        return 1

    def en_stack(self, bi, bj, bk, bl):
        return 1

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return 1

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 1

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 1

    def en_internal_init(self, sz):
        return 1

    def en_internal_asym(self, lup, rup):
        return 1


class RandomModel(Model):
    def __init__(self, seed=1):
        self.seed = seed

    def hash(self, *args):
        return float_hash(self.seed, *args)

    def en_ext_branch(self, bi, bj):
        return self.hash(bi, bj, 1)

    def en_multi_branch(self, bi, bk):
        return self.hash(bi, bk, 2)

    def en_5dangle(self, bim1, bi, bj):
        return self.hash(bim1, bi, bj, 3)

    def en_5dangle_inner(self, bi, bjm1, bj):
        return self.hash(bi, bjm1, bj, 4)

    def en_3dangle(self, bi, bj, bjp1):
        return self.hash(bi, bj, bjp1, 5)

    def en_3dangle_inner(self, bi, bip1, bj):
        return self.hash(bi, bip1, bj, 6)

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        return self.hash(bim1, bi, bj, bjp1, 7)

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        return self.hash(bi, bip1, bjm1, bj, 8)

    def en_multi_closing(self, bi, bj):
        return self.hash(bi, bj, 9)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return self.hash(bi, bj, bip1, bjm1, nunpaired, 10)

    def en_hairpin_special(self, id):
        return self.hash(id, 11)

    def en_stack(self, bi, bj, bk, bl):
        return self.hash(bi, bj, bk, bl, 12)

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return self.hash(bi, bj, bk, bl, nunpaired, 13)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return self.hash(bi, bj, bip1, bjm1, 14)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return self.hash(bi, bj, bim1, bjp1, 15)

    def en_internal_init(self, sz):
        return self.hash(sz, 16)

    def en_internal_asym(self, lup, rup):
        return self.hash(lup, rup, 17)


class RandomBulgeModel(RandomModel):
    def en_ext_branch(self, bi, bj):
        return 1

    def en_multi_branch(self, bi, bk):
        return 1

    def en_5dangle(self, bim1, bi, bj):
        return 1

    def en_5dangle_inner(self, bi, bjm1, bj):
        return 1

    def en_3dangle(self, bi, bj, bjp1):
        return 1

    def en_3dangle_inner(self, bi, bip1, bj):
        return 1

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        return 1

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        return 1

    def en_multi_closing(self, bi, bj):
        return 1

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 1

    def en_hairpin_special(self, id):
        return 1

    def en_stack(self, bi, bj, bk, bl):
        return 1

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 1

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 1

    def en_internal_init(self, sz):
        return 1

    def en_internal_asym(self, lup, rup):
        return 1



class NNModel(Model):

    def en_ext_branch(self, bi, bj):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0

        en = 0.0
        if pair in NON_GC_PAIRS:
            en += vienna_params['non_gc_closing_penalty']
        return boltz_onp(en)

    def en_multi_branch(self, bi, bk):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bk]
        if pair not in ALL_PAIRS:
            return 0.0

        dg = vienna_params['ml_branch']
        if pair in NON_GC_PAIRS:
            dg += vienna_params['non_gc_closing_penalty']

        return boltz_onp(dg)

    def en_5dangle(self, bim1, bi, bj):
        b_dangle5 = bim1
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0
        if b_dangle5 != INVALID_BASE:
            en = vienna_params['dangle5'][pair][RNA_ALPHA[b_dangle5]]
        else:
            en = 0.0
        return boltz_onp(en)

    def en_5dangle_inner(self, bi, bjm1, bj):
        return self.en_5dangle(bjm1, bj, bi)

    def en_3dangle(self, bi, bj, bjp1):
        b_dangle3 = bjp1
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0
        if b_dangle3 != INVALID_BASE:
            en = vienna_params['dangle3'][pair][RNA_ALPHA[b_dangle3]]
        else:
            en = 0.0
        return boltz_onp(en)

    def en_3dangle_inner(self, bi, bip1, bj):
        return self.en_3dangle(bj, bi, bip1)

    def _en_term_mismatch(self, bim1, bi, bj, bjp1):

        mm_table = vienna_params['mismatch_multi']
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0

        b_dangle5 = bim1
        b_dangle3 = bjp1
        if b_dangle5 != INVALID_BASE and b_dangle3 != INVALID_BASE:
            en = mm_table[pair][RNA_ALPHA[b_dangle5] + RNA_ALPHA[b_dangle3]]
        else:
            en = 0.0
        return en

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        en = self._en_term_mismatch(bim1, bi, bj, bjp1)
        return boltz_onp(en)

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        en = self._en_term_mismatch(bjm1, bj, bi, bip1)
        return boltz_onp(en)

    def en_multi_closing(self, bi, bj):
        closing_pair_dg = vienna_params['ml_initiation']
        closing_pair_dg += vienna_params['ml_branch']
        closing_pair_dg += non_gc_pairs_mat[bi, bj] * vienna_params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bj] * boltz_onp(closing_pair_dg)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        initiation = vienna_params['hairpin'][nunpaired] # FIXME: check against MAX_LOOP

        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if nunpaired == 3:
            non_gc_closing_penalty = 0
            if RNA_ALPHA[bi] + RNA_ALPHA[bj] in NON_GC_PAIRS:
                non_gc_closing_penalty = vienna_params['non_gc_closing_penalty']
            en = initiation + non_gc_closing_penalty
        else:
            mismatch = vienna_params['mismatch_hairpin'][pair][RNA_ALPHA[bip1]+RNA_ALPHA[bjm1]]
            en = initiation + mismatch

        return boltz_onp(en)

    def en_hairpin_special(self, id):
        # id is the index into SPECIAL_HAIRPINS
        en = special_hairpin_energies_onp[id]
        return boltz_onp(en)

    def _en_stack(self, bi, bj, bk, bl):
        pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
        en = vienna_params['stack'][pair1][pair2] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_onp(self._en_stack(bi, bj, bk, bl))

    def bulge_initiation(self, u):
        return vienna_params['bulge'][u] # FIXME: check against MAX_LOOP

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)
        if nunpaired == 1:
            bulge_dg += self._en_stack(bi, bj, bk, bl)
        else:
            pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
            if pair1 in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']

            pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
            if pair2 in NON_GC_PAIRS:
                bulge_dg += vienna_params['non_gc_closing_penalty']

        return boltz_onp(bulge_dg)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        en = vienna_params['mismatch_interior'][pair][RNA_ALPHA[bip1] + RNA_ALPHA[bjm1]]
        return boltz_onp(en)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        pair = RNA_ALPHA[bj] + RNA_ALPHA[bi]
        return boltz_onp(vienna_params['mismatch_interior'][pair][RNA_ALPHA[bjp1] + RNA_ALPHA[bim1]])

    def _en_internal_init(self, sz):
        return vienna_params['interior'][sz] # FIXME: check against MAX_LOOP
    def en_internal_init(self, sz):
        return boltz_onp(self._en_internal_init(sz))

    def _en_internal_asym(self, lup, rup):
        asym = onp.abs(lup-rup)
        dg = vienna_params['asymmetry'] * asym
        return onp.min([vienna_params['asymmetry_max'], dg])
    def en_internal_asym(self, lup, rup):
        return boltz_onp(self._en_internal_asym(lup, rup))

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        en = self.en_internal_init(lup+rup)*self.en_internal_asym(lup, rup)*self.en_il_inner_mismatch(
            bi, bj, bip1, bjm1)*self.en_il_outer_mismatch(bk, bl, bkm1, blp1)
        return en



class JaxNNModel(Model):

    def en_ext_branch(self, bi, bj):
        en = non_gc_pairs_mat[bi, bj] * vienna_params['non_gc_closing_penalty']
        return all_pairs_mat[bi, bj] * boltz_jnp(en)

    def en_multi_branch(self, bi, bk):
        dg = jax_vienna_params['ml_branch']
        dg += non_gc_pairs_mat[bi, bk] * vienna_params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bk] * boltz_jnp(dg)

    def en_5dangle(self, bim1, bi, bj):
        b_dangle5 = bim1
        en = jnp.where(b_dangle5 != INVALID_BASE,
                       jax_vienna_params['dangle5'][bi, bj, b_dangle5], 0.0)
        return boltz_jnp(en)

    def en_5dangle_inner(self, bi, bjm1, bj):
        return self.en_5dangle(bjm1, bj, bi)

    def en_3dangle(self, bi, bj, bjp1):
        b_dangle3 = bjp1
        en = jnp.where(b_dangle3 != INVALID_BASE,
                       jax_vienna_params['dangle3'][bi, bj, b_dangle3], 0.0)
        return boltz_jnp(en)

    def en_3dangle_inner(self, bi, bip1, bj):
        return self.en_3dangle(bj, bi, bip1)

    def _en_term_mismatch(self, bim1, bi, bj, bjp1):
        mm_table = jax_vienna_params['mismatch_multi']
        b_dangle5 = bim1
        b_dangle3 = bjp1
        both_dangles_cond = (b_dangle5 != INVALID_BASE) & (b_dangle3 != INVALID_BASE)
        en = jnp.where(both_dangles_cond, mm_table[bi, bj, b_dangle5, b_dangle3], 0.0)
        return en

    def en_term_mismatch(self, bim1, bi, bj, bjp1):
        en = self._en_term_mismatch(bim1, bi, bj, bjp1)
        return boltz_jnp(en)

    def en_term_mismatch_inner(self, bi, bip1, bjm1, bj):
        en = self._en_term_mismatch(bjm1, bj, bi, bip1)
        return boltz_jnp(en)

    def en_multi_closing(self, bi, bj):
        closing_pair_dg = jax_vienna_params['ml_initiation']
        closing_pair_dg += jax_vienna_params['ml_branch']
        closing_pair_dg += non_gc_pairs_mat[bi, bj] * vienna_params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bj] * boltz_jnp(closing_pair_dg)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        initiation = jax_vienna_params['hairpin'][nunpaired] # Note: nunpaired must be less than MAX_PRECOMPUTE

        # only used if u != 3
        mismatch = jax_vienna_params['mismatch_hairpin'][bi, bj, bip1, bjm1]

        # only used if u == 3
        non_gc_closing_penalty = non_gc_pairs_mat[bi, bj] * vienna_params['non_gc_closing_penalty']

        en = jnp.where(nunpaired == 3, initiation + non_gc_closing_penalty, initiation + mismatch)
        return boltz_jnp(en)

    def en_hairpin_special(self, id):
        # id is the index into SPECIAL_HAIRPINS
        en = special_hairpin_energies_jnp[id]
        return boltz_jnp(en)

    def _en_stack(self, bi, bj, bk, bl):
        en = jax_vienna_params['stack'][bi, bj, bl, bk] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_jnp(self._en_stack(bi, bj, bk, bl))

    def bulge_initiation(self, u):
        return jax_vienna_params['bulge'][u] # Note: u must be less than MAX_PRECOMPUTE

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)

        # Note: only used if nunpaired == 1
        stack_dg = self._en_stack(bi, bj, bk, bl)

        # Note: only used if nunpaired nunpaired != 1
        gc_penalty_dg = non_gc_pairs_mat[bi, bj] * vienna_params['non_gc_closing_penalty'] # FIXME: should these be jax_vienna_params even though it's a scalar?
        gc_penalty_dg += non_gc_pairs_mat[bl, bk] * vienna_params['non_gc_closing_penalty']

        bulge_dg += jnp.where(nunpaired == 1, stack_dg, gc_penalty_dg)
        return boltz_jnp(bulge_dg)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1,
                             mm_table=jax_vienna_params['mismatch_interior']):
        return boltz_jnp(mm_table[bi, bj, bip1, bjm1])

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1,
                             mm_table=jax_vienna_params['mismatch_interior']):
        return boltz_jnp(mm_table[bj, bi, bjp1, bim1])

    def _en_internal_init(self, sz):
        return jax_vienna_params['interior'][sz] # Note: sz must be less than MAX_PRECOMPUTE
    def en_internal_init(self, sz):
        return boltz_jnp(self._en_internal_init(sz))

    def _en_internal_asym(self, lup, rup):
        return jax_vienna_params['asymmetry_matrix'][lup, rup]
    def en_internal_asym(self, lup, rup):
        return boltz_jnp(self._en_internal_asym(lup, rup))

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        mm_table = jnp.where((lup == 1) | (rup == 1),
                             jax_vienna_params['mismatch_interior_1n'],
                             jnp.where(((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2)),
                                       jax_vienna_params['mismatch_interior_23'],
                                       jax_vienna_params['mismatch_interior']))

        gen_int = self.en_internal_init(lup+rup) * self.en_internal_asym(lup, rup) \
                  * self.en_il_inner_mismatch(bi, bj, bip1, bjm1, mm_table) \
                  * self.en_il_outer_mismatch(bk, bl, bkm1, blp1, mm_table)

        dg_boltz = jnp.where((lup == 1) & (rup == 1),
                       boltz_jnp(jax_vienna_params['int11'][bi, bj, bl, bk, bip1, bjm1]),
                       jnp.where((lup == 1) & (rup == 2),
                                 boltz_jnp(jax_vienna_params['int21'][bi, bj, bl, bk, bip1, blp1, bjm1]),
                                 jnp.where((lup == 2) & (rup == 1),
                                           boltz_jnp(jax_vienna_params['int21'][bl, bk, bi, bj, blp1, bip1, bkm1]),
                                           jnp.where((lup == 2) & (rup == 2),
                                                     boltz_jnp(jax_vienna_params['int22'][bi, bj, bl, bk, bip1, bkm1, blp1, bjm1]),
                                                     gen_int))))

        return dg_boltz



def dangle_dp(seq, branches, em: Model, closing_pair=None):
    n = len(branches)
    if n == 0:
        if closing_pair is None:
            return 1
        i, j = closing_pair
        en = 1
        if i+1 < j:
            en += em.en_3dangle_inner(seq[i], seq[i+1], seq[j])
            en += em.en_5dangle_inner(seq[i], seq[j-1], seq[j])
            if i+1 < j-1:
                en += em.en_term_mismatch_inner(seq[i], seq[i+1], seq[j-1], seq[j])
        return en
    branches = branches + \
        [(len(seq) if closing_pair is None else closing_pair[1], 0)]
    dp = onp.zeros((2, 2, n+1))
    dp[:, :, n] = 1
    for b in range(n-1, -1, -1):
        for last in range(2):
            for curr in range(2):
                i, j = branches[b]
                nexti = branches[b+1][0]
                dp[last, curr, b] = dp[last, int(nexti > j+1), b+1]
                if curr == 1:
                    dp[last, curr, b] += dp[last,
                                            int(nexti > j+1), b+1]*em.en_5dangle(seq[i-1], seq[i], seq[j])
                if b < n-1 or last:
                    if j+1 >= len(seq):
                        continue
                    if b < n-1 and nexti == j+1:
                        continue
                    dp[last, curr, b] += dp[last,
                                            int(nexti > j+2), b+1]*em.en_3dangle(seq[i], seq[j], seq[j+1])
                    if curr == 1:
                        dp[last, curr, b] += dp[last, int(nexti > j+2), b+1]*em.en_term_mismatch(
                            seq[i-1], seq[i], seq[j], seq[j+1])
    if closing_pair is None:
        return dp[int(branches[-2][1]+1 < len(seq)), int(branches[0][0] > 0), 0]
    else:
        i, j = closing_pair
        fi, fj = branches[0]
        li, lj = branches[-2]
        sm = dp[int(lj < j-1), int(i+1 < fi), 0]
        if i+1 < fi:
            sm += dp[int(lj < j-1), int(i+2 < fi), 0] * \
                em.en_3dangle_inner(seq[i], seq[i+1], seq[j])
        if lj < j-1:
            sm += dp[int(lj < j-2), int(i+1 < fi), 0] * \
                em.en_5dangle_inner(seq[i], seq[j-1], seq[j])
            if i+1 < fi:
                sm += dp[int(lj < j-2), int(i+2 < fi), 0] * \
                    em.en_term_mismatch_inner(
                        seq[i], seq[i+1], seq[j-1], seq[j])
        return sm


def calculate(str_seq, db, em: Model):
    seq = [RNA_ALPHA.index(c) for c in str_seq]

    ch, right = structure_tree(db)

    def calc_rec(atl):
        if atl == -1:
            sm = 1
            branches = []
            for cl in ch[atl]:
                sm *= calc_rec(cl)*em.en_ext_branch(seq[cl], seq[right[cl]])
                branches.append((cl, right[cl]))
            return sm*dangle_dp(seq, branches, em)
        if atl not in ch:
            s = str_seq[atl:right[atl]+1]
            idx = SPECIAL_HAIRPINS.index(s) if s in SPECIAL_HAIRPINS else -1
            return em.en_hairpin_special(idx) if idx != -1 else em.en_hairpin_not_special(
                seq[atl], seq[right[atl]], seq[atl+1], seq[right[atl]-1], right[atl]-atl-1)
        elif len(ch[atl]) == 1:
            cl, cr = ch[atl][0], right[ch[atl][0]]
            if cl == atl+1 and cr == right[atl]-1:
                return em.en_stack(seq[atl], seq[right[atl]], seq[cl], seq[cr])*calc_rec(cl)
            elif cl == atl+1 or cr == right[atl]-1:
                nunpaired = max(cl-atl-1, right[atl]-cr-1)
                return em.en_bulge(seq[atl], seq[right[atl]], seq[cl], seq[cr], nunpaired)*calc_rec(cl)
            else:
                bi = seq[atl]
                bj = seq[right[atl]]
                bip1 = seq[atl+1]
                bjm1 = seq[right[atl]-1]
                bk = seq[cl]
                bl = seq[cr]
                bkm1 = seq[cl-1]
                blp1 = seq[cr+1]
                lup = cl-atl-1
                rup = right[atl]-cr-1
                return em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*calc_rec(cl)
        else:
            sm = em.en_multi_closing(seq[atl], seq[right[atl]])
            branches = []
            for cl in ch[atl]:
                branches.append((cl, right[cl]))
                sm *= calc_rec(cl)*em.en_multi_branch(seq[cl], seq[right[cl]])
            return sm*dangle_dp(seq, branches, em, (atl, right[atl]))
    return calc_rec(-1)


beta = 1 / (kb*CELL_TEMP)
def dangle_dp_min(seq, branches, em: Model, closing_pair=None):
    n = len(branches)
    if n == 0:
        if closing_pair is None:
            return 1
        i, j = closing_pair
        en = 1
        dg = 0.0
        if i+1 < j:
            dangle3_inner_boltz = em.en_3dangle_inner(seq[i], seq[i+1], seq[j])
            dangle3_inner_dg = onp.log(dangle3_inner_boltz) * (-1/beta)
            if dangle3_inner_dg < dg:
                dg = dangle3_inner_dg
                en = dangle3_inner_boltz

            dangle5_inner_boltz = em.en_5dangle_inner(seq[i], seq[j-1], seq[j])
            dangle5_inner_dg = onp.log(dangle5_inner_boltz) * (-1/beta)
            if dangle5_inner_dg < dg:
                dg = dangle5_inner_dg
                en = dangle5_inner_boltz

            if i+1 < j-1:
                term_mismatch_inner_boltz = em.en_term_mismatch_inner(seq[i], seq[i+1], seq[j-1], seq[j])
                term_mismatch_inner_dg = onp.log(term_mismatch_inner_boltz) * (-1/beta)
                if term_mismatch_inner_dg < dg:
                    dg = term_mismatch_inner_dg
                    en = term_mismatch_inner_boltz
        return en
    branches = branches + \
        [(len(seq) if closing_pair is None else closing_pair[1], 0)]
    dp = onp.zeros((2, 2, n+1))
    dp[:, :, n] = 1
    for b in range(n-1, -1, -1):
        for last in range(2):
            for curr in range(2):
                i, j = branches[b]
                nexti = branches[b+1][0]
                dp[last, curr, b] = dp[last, int(nexti > j+1), b+1]
                current_dg = onp.log(dp[last, curr, b]) * (-1/beta)
                if curr == 1:
                    dangle5_boltz = dp[last, int(nexti > j+1), b+1] \
                                    * em.en_5dangle(seq[i-1], seq[i], seq[j])
                    dangle5_dg = onp.log(dangle5_boltz) * (-1/beta)
                    if dangle5_dg < current_dg:
                        dp[last, curr, b] = dangle5_boltz
                        current_dg = dangle5_dg

                if b < n-1 or last:
                    if j+1 >= len(seq):
                        continue
                    if b < n-1 and nexti == j+1:
                        continue

                    dangle3_boltz = dp[last, int(nexti > j+2), b+1] \
                                    * em.en_3dangle(seq[i], seq[j], seq[j+1])
                    dangle3_dg = onp.log(dangle3_boltz) * (-1/beta)
                    if dangle3_dg < current_dg:
                        dp[last, curr, b] = dangle3_boltz
                        current_dg = dangle3_dg
                    if curr == 1:
                        term_mismatch_boltz = dp[last, int(nexti > j+2), b+1] \
                                              * em.en_term_mismatch(seq[i-1], seq[i],
                                                                    seq[j], seq[j+1])
                        term_mismatch_dg = onp.log(term_mismatch_boltz) * (-1/beta)
                        if term_mismatch_dg < current_dg:
                            dp[last, curr, b] = term_mismatch_boltz
                            current_dg = term_mismatch_dg

    if closing_pair is None:
        return dp[int(branches[-2][1]+1 < len(seq)), int(branches[0][0] > 0), 0]
    else:
        i, j = closing_pair
        fi, fj = branches[0]
        li, lj = branches[-2]
        sm = dp[int(lj < j-1), int(i+1 < fi), 0]
        current_dg = onp.log(sm) * (-1/beta)
        if i+1 < fi:
            dangle3_inner_boltz = dp[int(lj < j-1), int(i+2 < fi), 0] * \
                                  em.en_3dangle_inner(seq[i], seq[i+1], seq[j])
            dangle3_inner_dg = onp.log(dangle3_inner_boltz) * (-1/beta)
            if dangle3_inner_dg < current_dg:
                sm = dangle3_inner_boltz
                current_dg = dangle3_inner_dg
        if lj < j-1:
            dangle5_inner_boltz = dp[int(lj < j-2), int(i+1 < fi), 0] * \
                                  em.en_5dangle_inner(seq[i], seq[j-1], seq[j])
            dangle5_inner_dg = onp.log(dangle5_inner_boltz) * (-1/beta)
            if dangle5_inner_dg < current_dg:
                sm = dangle5_inner_boltz
                current_dg = dangle5_inner_dg

            if i+1 < fi:
                term_mismatch_inner_boltz = dp[int(lj < j-2), int(i+2 < fi), 0] * \
                                            em.en_term_mismatch_inner(
                                                seq[i], seq[i+1], seq[j-1], seq[j])
                term_mismatch_inner_dg = onp.log(term_mismatch_inner_boltz) * (-1/beta)
                if term_mismatch_inner_dg < current_dg:
                    sm = term_mismatch_inner_boltz
                    current_dg = term_mismatch_inner_dg
        return sm



def calculate_min(str_seq, db, em: Model):
    seq = [RNA_ALPHA.index(c) for c in str_seq]

    ch, right = structure_tree(db)

    def calc_rec(atl):
        if atl == -1:
            sm = 1
            branches = []
            for cl in ch[atl]:
                ext_branch_val = em.en_ext_branch(seq[cl], seq[right[cl]])
                # print(f"External: {onp.log(ext_branch_val) * (-1/beta)}")
                sm *= calc_rec(cl)*ext_branch_val
                branches.append((cl, right[cl]))
            dp_val = dangle_dp_min(seq, branches, em)
            dp_dg = onp.log(dp_val) * (-1/beta)
            # print(f"External dp: {dp_dg}")
            return sm*dp_val
        if atl not in ch:
            s = str_seq[atl:right[atl]+1]
            idx = SPECIAL_HAIRPINS.index(s) if s in SPECIAL_HAIRPINS else -1
            hairpin_val = em.en_hairpin_special(idx) if idx != -1 else em.en_hairpin_not_special(
                seq[atl], seq[right[atl]], seq[atl+1], seq[right[atl]-1], right[atl]-atl-1)
            # print(atl)
            # print(right[atl])
            # print(onp.log(hairpin_val) * (-1/beta))
            return hairpin_val
        elif len(ch[atl]) == 1:
            cl, cr = ch[atl][0], right[ch[atl][0]]
            if cl == atl+1 and cr == right[atl]-1:
                return em.en_stack(seq[atl], seq[right[atl]], seq[cl], seq[cr])*calc_rec(cl)
            elif cl == atl+1 or cr == right[atl]-1:
                nunpaired = max(cl-atl-1, right[atl]-cr-1)
                return em.en_bulge(seq[atl], seq[right[atl]], seq[cl], seq[cr], nunpaired)*calc_rec(cl)
            else:
                bi = seq[atl]
                bj = seq[right[atl]]
                bip1 = seq[atl+1]
                bjm1 = seq[right[atl]-1]
                bk = seq[cl]
                bl = seq[cr]
                bkm1 = seq[cl-1]
                blp1 = seq[cr+1]
                lup = cl-atl-1
                rup = right[atl]-cr-1
                internal_val = em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)
                # print(f"Internal: {onp.log(internal_val) * (-1/beta)}")
                return internal_val*calc_rec(cl)
        else:
            sm = em.en_multi_closing(seq[atl], seq[right[atl]])
            branches = []
            for cl in ch[atl]:
                branches.append((cl, right[cl]))
                sm *= calc_rec(cl)*em.en_multi_branch(seq[cl], seq[right[cl]])
            return sm*dangle_dp_min(seq, branches, em, (atl, right[atl]))
    fin_sm = calc_rec(-1)
    return fin_sm

class TestEnergyCalculator(unittest.TestCase):

    def test_vienna(self):
        from jax_rnafold.common.utils import dot_bracket_2_matching, matching_2_dot_bracket
        from jax_rnafold.common.utils import seq_to_one_hot, get_rand_seq, random_pseq
        from jax_rnafold.common import vienna_rna, sampling
        import random
        from tqdm import tqdm

        n = 40
        max_structs = 50

        em = JaxNNModel()
        seq = get_rand_seq(n)
        # seq = "UCUGUCGACGGAGGGUUUAU"
        p_seq = jnp.array(seq_to_one_hot(seq))
        print(f"Sequence: {seq}")

        sampler = sampling.UniformStructureSampler()
        sampler.precomp(seq)
        n_structs = sampler.count_structures()
        if n_structs > max_structs:
            all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
        else:
            all_structs = [sampler.get_nth(i) for i in range(n_structs)]
        all_structs = [matching_2_dot_bracket(matching) for matching in all_structs]
        # all_structs = [".(.(....).)(...)...."]

        for db_str in tqdm(all_structs):
            print(f"\tStructure: {db_str}")

            matching = dot_bracket_2_matching(db_str)
            calc_boltz = calculate_min(seq, db_str, em)
            calc_dg = onp.log(calc_boltz) * (-1/beta)

            vienna_dg = vienna_rna.vienna_energy(seq, db_str, dangle_mode=1)
            print(f"\t\tCalculated dG: {calc_dg}")
            print(f"\t\tVienna dG: {vienna_dg}")

            # self.assertAlmostEqual(calc_dg, vienna_dg, places=7)
            self.assertAlmostEqual(calc_dg, vienna_dg, places=5)
