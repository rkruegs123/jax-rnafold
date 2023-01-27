import numpy as onp
import pdb
import functools
import unittest
import time

import jax
import optax
from jax import vmap, jit, grad, value_and_grad
from jax.tree_util import Partial
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import energy
from checkpoint import checkpoint_scan
from utils import bp_bases, HAIRPIN, N4, INVALID_BASE
from utils import SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_LENS, \
    SPECIAL_HAIRPIN_IDXS, N_SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_START_POS
from utils import matching_to_db
from utils import MAX_PRECOMPUTE
import brute_force
import nussinov as nus

import dp_discrete
from utils import get_rand_seq, seq_to_one_hot


f64 = jnp.float64

checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


# TODO (RK): save n, don't pass around everywhere else that we already are. Then, test lax.cond
def get_seq_partition_fn(em, db):
    n = len(db)

    match = [i for i in range(n+1)]
    stk = []
    for i in range(n):
        if db[i] == '(':
            stk.append(i)
        elif db[i] == ')':
            j = stk.pop()
            match[j+1] = i+1
    # Precomp unpaired ranges
    up = onp.ones((n+2, n+2), dtype=bool)
    for i in range(n+1):
        for j in range(i+2, n+1):
            up[i, j] = (db[j-2]=='.')*up[i, j-1]

    up = jnp.array(up)
    match = jnp.array(match, dtype=jnp.int64)

    def fill_external(i, padded_p_seq, n, P, E):
        j = match[i]
        i_end_cond = jnp.where(i == n, 1.0, 0.0)
        def bim1_bi_fn(bim1, bi):
            def bip1_fn(bip1):
                return (E[bi, bip1, i+1] + i_end_cond)*padded_p_seq[i+1, bip1]
            j_i_eq_val = jnp.sum(vmap(bip1_fn)(N4))

            def bj_bjp1_fn(bj, bjp1):
                j_end_cond = jnp.where(j == n, 1.0, 0.0)

                dangle5 = jnp.where(i == 1, INVALID_BASE, bim1) # FIXME: could be in outer scope
                dangle3 = jnp.where(j == n, INVALID_BASE, bjp1)
                return P[bi, bj, i, j] * (E[bj, bjp1, j+1] + j_end_cond) \
                    * em.en_ext_branch(dangle5, bi, bj, dangle3) \
                    * padded_p_seq[j, bj] * padded_p_seq[j+1, bjp1]
            all_bj_bjp1 = vmap(vmap(bj_bjp1_fn, (None, 0)), (0, None))
            j_i_neq_val = jnp.sum(all_bj_bjp1(N4, N4))

            return jnp.where(j == i, j_i_eq_val, j_i_neq_val)

        get_all_bim1_bi_sms = vmap(vmap(bim1_bi_fn, (None, 0)), (0, None)) # Note: I hope I got this ordering right. They are both (4,) so won't throw an error if wrong
        all_bim1_bi_sms = get_all_bim1_bi_sms(N4, N4)
        E = E.at[:, :, i].set(all_bim1_bi_sms)
        return E


    def fill_multi(i, padded_p_seq, n, ML, P):
        k = match[i]
        def get_inner_sm(bim1, bi, bj, bjp1, nb, j):

            def bip1_fn(bip1):
                correction_cond = (i+1 > j) & (nb == 0)
                correction = jnp.where(correction_cond, 1, 0)
                return (ML[bi, bip1, bj, bjp1, nb, i+1, j]+correction) * padded_p_seq[i+1, bip1]
            i_k_eq_val = jnp.sum(vmap(bip1_fn)(N4))

            nb_correction = jnp.where(nb <= 1, 1, 0)
            k_j_eq_val = P[bi, bj, i, j] * nb_correction \
                         * em.en_multi_branch(bim1, bi, bj, bjp1)

            def bk_fn(bk):
                return P[bi, bk, i, j-1] * nb_correction \
                    * em.en_multi_branch(bim1, bi, bk, bj) \
                    * padded_p_seq[j-1, bk]
            k_jm1_eq_val = jnp.sum(vmap(bk_fn)(N4))

            b_idx = jnp.where(nb-1 > 0, nb-1, 0) # branchless version of jnp.max(jnp.array([0, nb-1]))
            def bk_bkp1_fn(bk, bkp1):
                val = P[bi, bk, i, k] * ML[bk, bkp1, bj, bjp1, b_idx, k+1, j] \
                    * em.en_multi_branch(bim1, bi, bk, bkp1) \
                    * padded_p_seq[k, bk]*padded_p_seq[k+1, bkp1]
                return val
            bk_bkp1_fn = vmap(vmap(bk_bkp1_fn, (None, 0)), (0, None))
            gen_val = jnp.sum(bk_bkp1_fn(N4, N4))

            sm =  jnp.where(k == i, i_k_eq_val,
                            jnp.where(k == j, k_j_eq_val,
                                      jnp.where(k == j-1, k_jm1_eq_val, gen_val)))

            cond = (j >= i) & (j < n+1)
            return jnp.where(cond, sm, ML[bim1, bi, bj, bjp1, nb, i, j])

        get_all_inner_sms = vmap(get_inner_sm, (None, None, None, None, None, 0))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, None, None, 0, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, None, 0, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, 0, None, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, 0, None, None, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (0, None, None, None, None, None))
        js = jnp.arange(n+2)
        all_inner_sms = get_all_inner_sms(N4, N4, N4, N4, jnp.arange(3), js)
        ML = ML.at[:, :, :, :, :, i, :].set(all_inner_sms)
        return ML


    def fill_outer_mismatch(k, OMM, padded_p_seq, n):

        # For a fixed baes pair and mismatch, get the energy
        def get_bp_mm(bk, bl, bkm1, blp1, k, l):
            return em.en_il_outer_mismatch(bk, bl, bkm1, blp1) \
                * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1]

        # For a single l, get 6 values corresponding to the sum over each mismatch for each base pair
        def get_all_bp_all_mms(l):
            cond = (l >= k+1) & (l <= n)

            # For a fixed base pair, get the sum over all possible mismatches
            def get_bp_all_mms(bp):
                bk = bp[0]
                bl = bp[1]
                # Note: for now we do this with a vmap, but we could really do this with a tensor product and dot product
                mapped_fn = vmap(get_bp_mm, (None, None, None, 0, None, None))
                mapped_fn = vmap(mapped_fn, (None, None, 0, None, None, None))
                # mapped_fn = vmap(get_bp_mm, (None, None, 0, None, None, None))
                # mapped_fn = vmap(mapped_fn, (None, None, None, 0, None, None))
                all_mms = mapped_fn(bk, bl, N4, N4, k, l)
                # jax.debug.breakpoint()
                return jnp.sum(all_mms)

            true_val = vmap(get_bp_all_mms)(bp_bases)
            return jnp.where(cond, true_val, jnp.zeros(6)) # Note: set to 0 as we eventually *add* to OMM rather than *set*

        ls = jnp.arange(n+2)
        all_bp_mms = vmap(get_all_bp_all_mms)(ls)
        # OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T) # Note how we take the transpose here. Hope that is correct.
        OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T)
        return OMM



    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq, n):

        up2 = j-i+1
        u = j - i - 1

        # FIXME: repeated computation should combine with `psum_hairpin_special()`
        def pr_special_hairpin(id, i, j):
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            id_len = SPECIAL_HAIRPIN_LENS[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, SPECIAL_HAIRPIN_IDXS[idx_pos]], 1.0)
            ks = jnp.arange(n+1)
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        def special_hairpin_correction(id):
            sp_hairpin_len = SPECIAL_HAIRPIN_LENS[id]
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (SPECIAL_HAIRPIN_LENS[id] == up2) \
                       & (SPECIAL_HAIRPIN_IDXS[start_pos] == bi) \
                       & (SPECIAL_HAIRPIN_IDXS[end_pos] == bj)

            bjm1 = SPECIAL_HAIRPIN_IDXS[end_pos - 1]
            bip1 = SPECIAL_HAIRPIN_IDXS[start_pos + 1]
            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_not_special(
                                 bi, bj, bip1, bjm1, sp_hairpin_len - 2),
                             0.0)

        summands = vmap(special_hairpin_correction)(jnp.arange(N_SPECIAL_HAIRPINS))
        sm = jnp.sum(summands)
        return sm



    def psum_hairpin_special(bi, bj, i, j, padded_p_seq, n):

        up2 = j-i+1

        def pr_special_hairpin(id, i, j):
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            id_len = SPECIAL_HAIRPIN_LENS[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, SPECIAL_HAIRPIN_IDXS[idx_pos]], 1.0)
            ks = jnp.arange(n+1)
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        def special_hairpin(id):
            sp_hairpin_len = SPECIAL_HAIRPIN_LENS[id]
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            end_pos = start_pos + sp_hairpin_len - 1
            """
            id_valid = (SPECIAL_HAIRPINS[id][0] == RNA_ALPHA[bi]) \
                       & (SPECIAL_HAIRPINS[id][-1] == RNA_ALPHA[bj]) \
                       & (SPECIAL_HAIRPIN_LENS[id] == up2)
            """
            id_valid = (SPECIAL_HAIRPIN_LENS[id] == up2) \
                       & (SPECIAL_HAIRPIN_IDXS[start_pos] == bi) \
                       & (SPECIAL_HAIRPIN_IDXS[end_pos] == bj)

            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_special(id),
                             0.0)

        summands = vmap(special_hairpin)(jnp.arange(N_SPECIAL_HAIRPINS))
        sm = jnp.sum(summands)
        return sm


    def psum_hairpin_not_special(bi, bj, i, j, padded_p_seq, n):
        # Special case for HAIRPIN<=1
        # Necessary to respect conditional probability the mismatch

        u = j-i-1

        def u1_fn(bip1):
            return padded_p_seq[i+1, bip1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bip1, 1)
        u1_fn = vmap(u1_fn)

        def u_general_fn(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bjm1, j-i-1)
        u_general_fn = vmap(vmap(u_general_fn, (0, None)), (None, 0))


        return jnp.where(u == 0, em.en_hairpin_not_special(bi, bj, bj, bi, 0),
                         jnp.where(u == 1,
                                   jnp.sum(u1_fn(N4)),
                                   jnp.sum(u_general_fn(N4, N4))))


    # Note: all hairpin operations the exact same as for ss_partition
    def psum_hairpin(bi, bj, i, j, padded_p_seq, n):
        return psum_hairpin_not_special(bi, bj, i, j, padded_p_seq, n) \
            + psum_hairpin_special(bi, bj, i, j, padded_p_seq, n) \
            - psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq, n)

    def psum_bulges(bi, bj, i, j, padded_p_seq, n, P):

        def get_bp_kl(bp, kl):
            cond = (kl >= i+2) & (kl < j-1)
            bp_kl_sm = 0
            bk = bp[0]
            bl = bp[1]

            bp_kl_sm += P[bk, bl, i+1, kl]*padded_p_seq[i+1, bk] * \
                padded_p_seq[kl, bl]*em.en_bulge(bi, bj, bk, bl, j-kl-1)*up[kl, j] # note the lookpu in `up`
            bp_kl_sm += P[bk, bl, kl, j-1]*padded_p_seq[kl, bk] * \
                padded_p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, kl-i-1)*up[i, kl] # note the lookup in `up`
            return jnp.where(cond, bp_kl_sm, 0.0) # default is 0.0 because we will sum

        def get_bp_all_kl(bp):
            all_kls = jnp.arange(n+1)
            all_bp_kl_sms = vmap(get_bp_kl, (None, 0))(bp, all_kls)
            return jnp.sum(all_bp_kl_sms)

        all_bp_sms = vmap(get_bp_all_kl)(bp_bases)
        return jnp.sum(all_bp_sms)


    # Note: exact same as for sequence-structure PF
    def psum_internal_loops(bi, bj, i, j, padded_p_seq, n, P, OMM):
        def get_mmij_term(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
        mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
        mmij = jnp.sum(mmij_terms)

        # Note: 1x1 and 1xN and Nx1. Not just 1xN.
        def get_bp_1n_sm(bp, bip1, bjm1):
            bk = bp[0]
            bl = bp[1]

            # 1x1
            pr_ij_mm = padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1]
            bp_1n_sm = P[bk, bl, i+2, j-2] * padded_p_seq[i+2, bk] * padded_p_seq[j-2, bl] \
                       * pr_ij_mm \
                       * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bip1, bjm1, 1, 1) \
                       * up[i, i+2] * up[j-2, j]

            def get_z_b_sm(z, b):
                zb_sm = 0.0
                il_en = em.en_internal(
                    bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-z-1)
                zb_sm += P[bk, bl, i+2, z]*padded_p_seq[i+2, bk] \
                    * padded_p_seq[z, bl]*padded_p_seq[z+1, b]*pr_ij_mm*il_en \
                    * up[i, i+2] * up[z, j]
                il_en = em.en_internal(
                    bi, bj, bk, bl, bip1, bjm1, b, bjm1, z-i-1, 1)
                zb_sm += P[bk, bl, z, j-2]*padded_p_seq[z, bk] \
                    * padded_p_seq[j-2, bl]*padded_p_seq[z-1, b]*pr_ij_mm*il_en \
                    * up[i, z] * up[j-2, j]
                return zb_sm

            def get_z_all_bs_sm(z):
                all_bs_summands = vmap(get_z_b_sm, (None, 0))(z, N4)
                all_bs_sm = jnp.sum(all_bs_summands)

                cond = (z >= i+3) & (z < j-2)
                return jnp.where(cond, all_bs_sm, 0.0)

            zs = jnp.arange(n+2)
            all_zs_sms = vmap(get_z_all_bs_sm)(zs)
            bp_1n_sm += jnp.sum(all_zs_sms)
            return bp_1n_sm


        def get_bp_special_sm(bp, k, l, bk, bl, lup, rup):
            def get_bp_special_summand(bip1, bjm1, bkm1, blp1):
                return P[bk, bl, k, l] * padded_p_seq[k, bk] * padded_p_seq[l, bl] \
                    * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) \
                    * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1] * padded_p_seq[j-1, bjm1] \
                    * up[i, k] * up[l, j]
            get_all_summands = vmap(get_bp_special_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))
            all_summands = get_all_summands(N4, N4, N4, N4)
            return jnp.sum(all_summands)

        def get_bp_general_sm(bp, k):
            bk = bp[0]
            bl = bp[1]
            l = match[k]
            idx_cond = (k >= i+2) & (k < j-2)
            lup = k-i-1
            rup = j-l-1
            n1_cond = (lup > 1) & (rup > 1)
            actually_paired_cond = l > k
            cond = idx_cond & n1_cond & actually_paired_cond

            init_and_pair = em.en_internal_init(lup+rup) \
                            * em.en_internal_asym(lup, rup) \
                            * P[bk, bl, k, l] \
                            * padded_p_seq[k, bk]*padded_p_seq[l, bl]
            gen_sm = OMM[bk, bl, k, l]*mmij*init_and_pair*up[i, k]*up[l, j]

            is_special = ((lup == 2) & (rup == 2)) \
                         | ((lup == 2) & (rup == 3)) \
                         | ((lup == 3) & (rup == 2))

            return jnp.where(cond,
                             jnp.where(is_special,
                                       get_bp_special_sm(bp, k, l, bk, bl, lup, rup),
                                       gen_sm),
                             0.0)

        def get_bp_sm(bp):
            bk = bp[0]
            bl = bp[1]
            bp_sum = 0.0

            all_1n_sms = vmap(vmap(get_bp_1n_sm, (None, 0, None)), (None, None, 0))(bp, N4, N4)
            bp_sum += jnp.sum(all_1n_sms)

            ks = jnp.arange(n+2)
            # all_gen_sms = vmap(vmap(get_bp_general_sm, (None, 0, None)), (None, None, 0))(bp, ks, ls)
            all_gen_sms = vmap(get_bp_general_sm, (None, 0))(bp, ks)
            bp_sum += jnp.sum(all_gen_sms)

            return bp_sum

        # vmap over get_bp_sm, and sum all results
        all_bp_sms = vmap(get_bp_sm)(bp_bases)
        return jnp.sum(all_bp_sms)

    def _fill_paired(i, padded_p_seq, n, OMM, ML, P):
        j = match[i]

        def get_bp_stack(bp, j, bi, bj):
            bk = bp[0]
            bl = bp[1]
            return P[bk, bl, i+1, j-1]*padded_p_seq[i+1, bk] * \
                padded_p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)

        def get_ml(bip1, bjm1, j, bi, bj):
            return ML[bi, bip1, bjm1, bj, 2, i+1, j-1]*padded_p_seq[i+1, bip1] * \
                padded_p_seq[j-1, bjm1] * \
                em.en_multi_closing(bi, bip1, bjm1, bj)


        # For a given bp, get the corresponding sum
        def get_bp_sm(bp):
            bi = bp[0]
            bj = bp[1]
            sm = psum_hairpin(bi, bj, i, j, padded_p_seq, n) * up[i, j]
            sm += psum_bulges(bi, bj, i, j, padded_p_seq, n, P)
            sm += psum_internal_loops(bi, bj, i, j, padded_p_seq, n, P, OMM)

            # Stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(bp_bases, j, bi, bj)
            sm += jnp.sum(stack_summands)

            # Multiloops
            ml_fn = vmap(get_ml, (0, None, None, None, None))
            ml_fn = vmap(ml_fn, (None, 0, None, None, None))
            ml_summands = ml_fn(N4, N4, j, bi, bj)
            sm += jnp.sum(ml_summands)

            cond = (j >= i+HAIRPIN+1) & (j < n+1)
            return jnp.where(cond, sm, P[bi, bj, i, j])

        # For our fixed j, get all sums for each bp
        all_bp_sms = vmap(get_bp_sm)(bp_bases)
        P = P.at[bp_bases[:, 0], bp_bases[:, 1], i, j].set(all_bp_sms)
        return P

    def fill_paired(i, padded_p_seq, n, OMM, ML, P):
        j = match[i]
        return jnp.where(i == j, P, _fill_paired(i, padded_p_seq, n, OMM, ML, P))

        """
        return jax.lax.cond(j == i,
                            lambda v1, v2, v3, v4, v5, v6: P,
                            _fill_paired,
                            i, padded_p_seq, n, OMM, ML, P)
        """


    def seq_partition(p_seq):
        n = p_seq.shape[0]

        # Pad appropriately
        padded_p_seq = jnp.zeros((n+2, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[1:n+1].set(p_seq)
        padded_p_seq = padded_p_seq.at[0, 0].set(1.0)
        padded_p_seq = padded_p_seq.at[-1, 0].set(1.0)

        E = jnp.zeros((4, 4, n+2), dtype=f64)
        P = jnp.zeros((4, 4, n+2, n+2), dtype=f64)
        ML = jnp.zeros((4, 4, 4, 4, 3, n+2, n+2), dtype=f64)
        OMM = jnp.zeros((4, 4, n+2, n+2), dtype=f64)

        def fill_table(carry, i):
            OMM, P, ML, E = carry

            OMM = fill_outer_mismatch(i, OMM, padded_p_seq, n)
            P = fill_paired(i, padded_p_seq, n, OMM, ML, P)
            ML = fill_multi(i, padded_p_seq, n, ML, P)
            E = fill_external(i, padded_p_seq, n, P, E)

            return (OMM, P, ML, E), None

        (OMM, P, ML, E), _ = scan(fill_table,
                                  (OMM, P, ML, E),
                                  jnp.arange(n, 0, -1))

        def get_fin_summand(bim1, bi):
            return E[bim1, bi, 1]*padded_p_seq[1, bi]*padded_p_seq[0, bim1]
        get_all_fin_summands = vmap(vmap(get_fin_summand, (None, 0)), (0, None))
        all_fin_summands = get_all_fin_summands(N4, N4)
        fin_sm = jnp.sum(all_fin_summands)
        return fin_sm

    return seq_partition


def fuzz_test(n, num_seq, em, tol=1e-6, max_structs=20):
    import utils
    from tqdm import tqdm
    import random

    import vienna_rna
    import sampling


    seqs = [utils.get_rand_seq(n) for _ in range(num_seq)]

    failed_cases = list()
    n_passed = 0

    for seq in seqs:
        p_seq = jnp.array(seq_to_one_hot(seq))

        print(f"Sequence: {seq}")
        sampler = sampling.UniformStructureSampler()
        sampler.precomp(seq)
        n_structs = sampler.count_structures()
        if n_structs > max_structs:
            all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
        else:
            all_structs = [sampler.get_nth(i) for i in range(n_structs)]
        all_structs = [utils.matching_2_dot_bracket(matching) for matching in all_structs]

        print(f"Found {len(all_structs)} structures")

        for db_str in tqdm(all_structs):
            seq_partition_fn = get_seq_partition_fn(em, db_str)

            print(f"\n\tStructure: {db_str}")

            reference_seq_pf = energy.calculate(seq, db_str, em)
            print(f"\t\tReference Seq PF: {reference_seq_pf}")

            seq_pf = seq_partition_fn(p_seq)
            print(f"\t\tOur Seq PF: {seq_pf}")

            if onp.abs(seq_pf - reference_seq_pf) > tol:
                failed_cases.append((seq, db_str, reference_seq_pf, seq_pf))
                print(utils.bcolors.FAIL + "\t\tFail!\n" + utils.bcolors.ENDC)
                pdb.set_trace()
            else:
                print(utils.bcolors.OKGREEN + "\t\tSuccess!\n" + utils.bcolors.ENDC)
                n_passed += 1
    if not failed_cases:
        print(f"\nAll tests passed!")
    else:
        print(f"\nFailed tests:")
        for seq, struct, reference_seq_pf, seq_pf in failed_cases:
            print(f"- {seq}, {struct} -- {reference_seq_pf} (Reference) vs. {seq_pf}")



if __name__ == "__main__":
    import vienna

    em = energy.JaxNNModel()
    # em = energy.RandomHairpinModel()
    # em = energy.RandomILModel()
    # em = energy.RandomBulgeModel()
    # em = energy.RandomMultiloopModel()

    fuzz_test(n=24, num_seq=10, em=em, tol=1e-6)
    pdb.set_trace()


    # seq = "GGAAACGAAACC"
    # db_str = "((...)(...))" # simple multiloop

    seq = "CAUACAGGUUUAGUAAUUGGC"
    db_str = "((.(....).).)(....).."


    p_seq = jnp.array(seq_to_one_hot(seq))

    em = energy.RandomILModel()


    seq_partition_fn = get_seq_partition_fn(em, db_str)
    start = time.time()
    seq_pf = seq_partition_fn(p_seq)
    end = time.time()
    print(f"Our seq pf: {seq_pf}")

    reference_discrete_seq_pf = energy.calculate(seq, db_str, em)
    print(f"Reference discrete seq pf: {reference_discrete_seq_pf}")

    # reference_seq_pf = vienna.seq_partition(p_seq, db_str, em)
    # print(f"Reference seq pf: {reference_discrete_seq_pf}")

    pdb.set_trace()
