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

from jax_rnafold.common.checkpoint import checkpoint_scan
from jax_rnafold.common.utils import bp_bases, HAIRPIN, N4, INVALID_BASE
from jax_rnafold.common.utils import SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_LENS, \
    SPECIAL_HAIRPIN_IDXS, N_SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_START_POS
from jax_rnafold.common.utils import matching_to_db
from jax_rnafold.common.utils import MAX_PRECOMPUTE, MAX_LOOP
from jax_rnafold.common import brute_force
from jax_rnafold.common import nussinov as nus
from jax_rnafold.common.utils import get_rand_seq, seq_to_one_hot

from jax_rnafold.d2 import energy
from jax_rnafold.d2 import dp_discrete
from jax_rnafold.d2 import reference



f64 = jnp.float64

checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


# TODO: remove all unnecessary n parameterizations
def get_ss_partition_fn(em, seq_len, max_loop=MAX_LOOP):
    two_loop_length = min(seq_len, max_loop)

    @jit
    def fill_outer_mismatch(k, OMM, padded_p_seq):

        # For a fixed baes pair and mismatch, get the energy
        def get_bp_mm(bk, bl, bkm1, blp1, k, l):
            return em.en_il_outer_mismatch(bk, bl, bkm1, blp1) \
                * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1]

        # For a single l, get 6 values corresponding to the sum over each mismatch for each base pair
        def get_all_bp_all_mms(l):
            cond = (l >= k+1) & (l <= seq_len)

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

        ls = jnp.arange(seq_len+2)
        all_bp_mms = vmap(get_all_bp_all_mms)(ls)
        # OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T) # Note how we take the transpose here.
        OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T)
        return OMM


    @jit
    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq):

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
            ks = jnp.arange(seq_len+1) # FIXME: is this correct?
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


    @jit
    def psum_hairpin_special(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1

        def pr_special_hairpin(id, i, j):
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            id_len = SPECIAL_HAIRPIN_LENS[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, SPECIAL_HAIRPIN_IDXS[idx_pos]], 1.0)
            ks = jnp.arange(seq_len+1) # FIXME: is this correct?
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        def special_hairpin(id):
            sp_hairpin_len = SPECIAL_HAIRPIN_LENS[id]
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (SPECIAL_HAIRPIN_LENS[id] == up2) \
                       & (SPECIAL_HAIRPIN_IDXS[start_pos] == bi) \
                       & (SPECIAL_HAIRPIN_IDXS[end_pos] == bj)

            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_special(id),
                             0.0)

        summands = vmap(special_hairpin)(jnp.arange(N_SPECIAL_HAIRPINS))
        sm = jnp.sum(summands)
        return sm


    @jit
    def psum_hairpin_not_special(bi, bj, i, j, padded_p_seq):
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


    @jit
    def psum_hairpin(bi, bj, i, j, padded_p_seq):
        return psum_hairpin_not_special(bi, bj, i, j, padded_p_seq) \
            + psum_hairpin_special(bi, bj, i, j, padded_p_seq) \
            - psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq)

    @jit
    def psum_bulges(bi, bj, i, j, padded_p_seq, P):

        def get_bp_kl(bp, kl_offset):
            kl = kl_offset + i + 2
            cond = (kl >= i+2) & (kl < j-1)
            bp_kl_sm = 0
            bk = bp[0]
            bl = bp[1]

            bp_kl_sm += P[bk, bl, i+1, kl]*padded_p_seq[i+1, bk] * \
                padded_p_seq[kl, bl]*em.en_bulge(bi, bj, bk, bl, j-kl-1)
            bp_kl_sm += P[bk, bl, kl, j-1]*padded_p_seq[kl, bk] * \
                padded_p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, kl-i-1)
            return jnp.where(cond, bp_kl_sm, 0.0) # default is 0.0 because we will sum

        def get_bp_all_kl(bp):
            # all_kls = jnp.arange(n+1) # FIXME: is this the appropriate size? Will we be missing anything?
            # all_kls = jnp.arange(i+2, i+2+two_loop_length)
            all_kl_offsets = jnp.arange(two_loop_length)
            # all_bp_kl_sms = vmap(get_bp_kl, (None, 0))(bp, all_kls)
            all_bp_kl_sms = vmap(get_bp_kl, (None, 0))(bp, all_kl_offsets)
            return jnp.sum(all_bp_kl_sms)

        all_bp_sms = vmap(get_bp_all_kl)(bp_bases)
        return jnp.sum(all_bp_sms)

    @jit
    def psum_internal_loops(bi, bj, i, j, padded_p_seq, P, OMM):
        def get_mmij_term(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
        mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
        mmij = jnp.sum(mmij_terms)

        # Note: 1x1 and 1xN and Nx1. Not just 1xN.
        def get_bp_1n_sm(bp, bip1, bjm1):
            bk = bp[0]
            bl = bp[1]

            pr_ij_mm = padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1]
            bp_1n_sm = P[bk, bl, i+2, j-2]*padded_p_seq[i+2, bk]*padded_p_seq[j-2, bl] * \
                       pr_ij_mm * em.en_internal(bi, bj, bk, bl,
                                                 bip1, bjm1, bip1, bjm1, 1, 1)

            def get_z_b_sm(z, b):
                zb_sm = 0.0
                il_en = em.en_internal(
                    bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-z-1)
                zb_sm += P[bk, bl, i+2, z]*padded_p_seq[i+2, bk] * \
                    padded_p_seq[z, bl]*padded_p_seq[z+1, b]*pr_ij_mm*il_en
                il_en = em.en_internal(
                    bi, bj, bk, bl, bip1, bjm1, b, bjm1, z-i-1, 1)
                zb_sm += P[bk, bl, z, j-2]*padded_p_seq[z, bk] * \
                    padded_p_seq[j-2, bl]*padded_p_seq[z-1, b]*pr_ij_mm*il_en
                return zb_sm

            def get_z_all_bs_sm(z_offset):
                z = z_offset + i + 3
                all_bs_summands = vmap(get_z_b_sm, (None, 0))(z, N4)
                all_bs_sm = jnp.sum(all_bs_summands)

                cond = (z >= i+3) & (z < j-2)
                return jnp.where(cond, all_bs_sm, 0.0)

            # zs = jnp.arange(n+2) # FIXME: is this the right range? Does it miss anything?
            # all_zs_sms = vmap(get_z_all_bs_sm)(zs)
            z_offsets = jnp.arange(two_loop_length)
            all_zs_sms = vmap(get_z_all_bs_sm)(z_offsets)

            bp_1n_sm += jnp.sum(all_zs_sms)
            return bp_1n_sm


        def get_bp_special_sm(bp, k, l, bk, bl, lup, rup):
            def get_bp_special_summand(bip1, bjm1, bkm1, blp1):
                return P[bk, bl, k, l] * padded_p_seq[k, bk] * padded_p_seq[l, bl] \
                    * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) \
                    * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1] * padded_p_seq[j-1, bjm1]
            get_all_summands = vmap(get_bp_special_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))
            all_summands = get_all_summands(N4, N4, N4, N4)
            return jnp.sum(all_summands)

        def get_bp_general_sm(bp, k_offset, l_offset):
            k = k_offset + i + 2
            l = l_offset + k + 1 # note: uses the k from above that includes k_offset. Order matters here.

            bk = bp[0]
            bl = bp[1]
            idx_cond = (k >= i+2) & (k < j-2) & (l >= k+1) & (l < j-1)
            lup = k-i-1
            rup = j-l-1
            n1_cond = (lup > 1) & (rup > 1)
            cond = idx_cond & n1_cond

            init_and_pair = em.en_internal_init(lup+rup) \
                            * em.en_internal_asym(lup, rup) \
                            * P[bk, bl, k, l] \
                            * padded_p_seq[k, bk]*padded_p_seq[l, bl]
            gen_sm = OMM[bk, bl, k, l]*mmij*init_and_pair

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

            # ks = jnp.arange(n+2) # FIXME: is this correct?
            # ls = jnp.arange(n+2) # FIXME: is this correct?
            k_offsets = jnp.arange(two_loop_length)
            l_offsets = jnp.arange(two_loop_length)
            all_gen_sms = vmap(vmap(get_bp_general_sm, (None, 0, None)), (None, None, 0))(bp, k_offsets, l_offsets)
            bp_sum += jnp.sum(all_gen_sms)

            return bp_sum

        # vmap over get_bp_sm, and sum all results
        all_bp_sms = vmap(get_bp_sm)(bp_bases)
        return jnp.sum(all_bp_sms)


    @jit
    def fill_paired(i, padded_p_seq, OMM, ML, P):

        def get_bp_stack(bp, j, bi, bj):
            bk = bp[0]
            bl = bp[1]
            return P[bk, bl, i+1, j-1]*padded_p_seq[i+1, bk] * \
                padded_p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)

        def get_ml(bip1, bjm1, j, bi, bj):
            return ML[bi, bip1, bjm1, bj, 2, i+1, j-1]*padded_p_seq[i+1, bip1] * \
                padded_p_seq[j-1, bjm1] * \
                em.en_multi_closing(bi, bip1, bjm1, bj)

        # For a given bp and j, get the corresponding sum
        def get_bp_j_sm(bp, j):
            bi = bp[0]
            bj = bp[1]
            sm = psum_hairpin(bi, bj, i, j, padded_p_seq)
            sm += psum_bulges(bi, bj, i, j, padded_p_seq, P)
            sm += psum_internal_loops(bi, bj, i, j, padded_p_seq, P, OMM)

            # Stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(bp_bases, j, bi, bj)
            sm += jnp.sum(stack_summands)

            # Multiloops
            ml_fn = vmap(get_ml, (0, None, None, None, None))
            ml_fn = vmap(ml_fn, (None, 0, None, None, None))
            ml_summands = ml_fn(N4, N4, j, bi, bj)
            sm += jnp.sum(ml_summands)


            cond = (j >= i+HAIRPIN+1) & (j < seq_len+1)
            return jnp.where(cond, sm, P[bi, bj, i, j])


        # For a fixed base pair, get all js
        def get_bp_all_js(bp):
            js = jnp.arange(seq_len+2)
            return vmap(get_bp_j_sm, (None, 0))(bp, js)

        all_bp_js = vmap(get_bp_all_js)(bp_bases)
        P = P.at[bp_bases[:, 0], bp_bases[:, 1], i].set(all_bp_js) # Note how we *dont* take the transpose here
        return P


    @jit
    def fill_multi(i, padded_p_seq, ML, P):
        def get_inner_sm(bim1, bi, bj, bjp1, nb, j):
            sm = 0.0

            def bip1_fn(bip1):
                correction_cond = (i+1 > j) & (nb == 0)
                correction = jnp.where(correction_cond, 1, 0)
                return (ML[bi, bip1, bj, bjp1, nb, i+1, j]+correction) * padded_p_seq[i+1, bip1]
            sm += jnp.sum(vmap(bip1_fn)(N4))

            special_kj_bit = jnp.where(nb <= 1, 1, 0) # not sure if int() casting is diff. in JAX
            sm += P[bi, bj, i, j]*special_kj_bit * \
                  em.en_multi_branch(bim1, bi, bj, bjp1)

            def bk_fn(bk):
                correction = jnp.where(nb <= 1, 1, 0)
                return P[bi, bk, i, j-1] * correction \
                    * em.en_multi_branch(bim1, bi, bk, bj) * padded_p_seq[j-1, bk]
            sm += jnp.sum(vmap(bk_fn)(N4))

            def k_bk_bkp1_fn(k, bk, bkp1):
                b_idx = jnp.where(nb-1 > 0, nb-1, 0) # branchless version of jnp.max(jnp.array([0, nb-1]))
                # b_idx = jnp.max(jnp.array([0, nb-1]))

                val = P[bi, bk, i, k] * ML[bk, bkp1, bj, bjp1, b_idx, k+1, j] \
                    * em.en_multi_branch(bim1, bi, bk, bkp1) \
                    * padded_p_seq[k, bk]*padded_p_seq[k+1, bkp1]
                cond = (k >= i) & (k < j-1)
                return jnp.where(cond, val, 0.0)
            k_bk_bkp1_fn = vmap(k_bk_bkp1_fn, (0, None, None))
            k_bk_bkp1_fn = vmap(k_bk_bkp1_fn, (None, 0, None))
            k_bk_bkp1_fn = vmap(k_bk_bkp1_fn, (None, None, 0))
            ks = jnp.arange(seq_len+2) # FIXME: is this correct?
            sm += jnp.sum(k_bk_bkp1_fn(ks, N4, N4))

            cond = (j >= i) & (j < seq_len+1)
            return jnp.where(cond, sm, ML[bim1, bi, bj, bjp1, nb, i, j])
        # Note: we do the reverse order of in_axes here rather than taking the transpose. Could use elsewhere.

        get_all_inner_sms = vmap(get_inner_sm, (None, None, None, None, None, 0))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, None, None, 0, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, None, 0, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, None, 0, None, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (None, 0, None, None, None, None))
        get_all_inner_sms = vmap(get_all_inner_sms, (0, None, None, None, None, None))
        js = jnp.arange(seq_len+2) # FIXME: is this correct?
        all_inner_sms = get_all_inner_sms(N4, N4, N4, N4, jnp.arange(3), js)
        ML = ML.at[:, :, :, :, :, i, :].set(all_inner_sms)
        return ML



    @jit
    def fill_external(i, padded_p_seq, P, E):
        def bim1_bi_fn(bim1, bi):
            def bip1_fn(bip1):
                end_correction = jnp.where(i == seq_len, 1, 0)
                return (E[bi, bip1, i+1] + end_correction)*padded_p_seq[i+1, bip1]
            sm = jnp.sum(vmap(bip1_fn)(N4))

            def j_bj_bjp1_fn(j, bj, bjp1):
                dangle5 = jnp.where(i == 1, INVALID_BASE, bim1)
                dangle3 = jnp.where(j == seq_len, INVALID_BASE, bjp1)

                end_correction = jnp.where(j == seq_len, 1.0, 0.0)
                val = P[bi, bj, i, j] * (E[bj, bjp1, j+1] + end_correction) \
                      * em.en_ext_branch(dangle5, bi, bj, dangle3) \
                      * padded_p_seq[j, bj] * padded_p_seq[j+1, bjp1]

                cond = (j >= i+1) & (j < seq_len+1)
                return jnp.where(cond, val, 0.0)
            all_j_bj_bjp1_fn = vmap(vmap(vmap(j_bj_bjp1_fn, (None, None, 0)),
                                         (None, 0, None)),
                                    (0, None, None))
            js = jnp.arange(seq_len+1)
            sm += jnp.sum(all_j_bj_bjp1_fn(js, N4, N4))
            return sm

        get_all_bim1_bi_sms = vmap(vmap(bim1_bi_fn, (None, 0)), (0, None)) # Note: Have to be careful about getting ordering right. They are both (4,) so won't throw an error if wrong
        all_bim1_bi_sms = get_all_bim1_bi_sms(N4, N4)
        E = E.at[:, :, i].set(all_bim1_bi_sms)
        return E


    def ss_partition(p_seq):

        # Pad appropriately
        padded_p_seq = jnp.zeros((seq_len+2, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[1:seq_len+1].set(p_seq)
        padded_p_seq = padded_p_seq.at[0, 0].set(1.0)
        padded_p_seq = padded_p_seq.at[-1, 0].set(1.0)

        E = jnp.zeros((4, 4, seq_len+2), dtype=f64)
        P = jnp.zeros((4, 4, seq_len+2, seq_len+2), dtype=f64)
        ML = jnp.zeros((4, 4, 4, 4, 3, seq_len+2, seq_len+2), dtype=f64)
        OMM = jnp.zeros((4, 4, seq_len+2, seq_len+2), dtype=f64)

        @jit
        def fill_table(carry, i):
            OMM, P, ML, E = carry

            OMM = fill_outer_mismatch(i, OMM, padded_p_seq)
            P = fill_paired(i, padded_p_seq, OMM, ML, P)
            ML = fill_multi(i, padded_p_seq, ML, P)
            E = fill_external(i, padded_p_seq, P, E)

            return (OMM, P, ML, E), None

        (OMM, P, ML, E), _ = scan(fill_table,
                                  (OMM, P, ML, E),
                                  jnp.arange(seq_len, 0, -1))

        def get_fin_summand(bim1, bi):
            return E[bim1, bi, 1]*padded_p_seq[1, bi]*padded_p_seq[0, bim1]
        get_all_fin_summands = vmap(vmap(get_fin_summand, (None, 0)), (0, None))
        all_fin_summands = get_all_fin_summands(N4, N4)
        fin_sm = jnp.sum(all_fin_summands)
        return fin_sm

    return ss_partition



class TestPartitionFunction(unittest.TestCase):
    def _random_p_seq(self, n):
        p_seq = onp.empty((n, 4), dtype=onp.float64)
        for i in range(n):
            p_seq[i] = onp.random.random_sample(4)
            p_seq[i] /= onp.sum(p_seq[i])
        return p_seq

    def _all_1_test(self, n):
        em = energy.All1Model()
        ss_partition_fn = get_ss_partition_fn(em)

        p_seq = self._random_p_seq(n)
        nuss = nus.ss_partition(p_seq, en_pair=nus.en_pair_1)
        vien = ss_partition_fn(p_seq)
        print(n, nuss, vien)
        self.assertAlmostEqual(nuss, vien, places=7)

    def _brute_model_test(self, em, n):
        p_seq = self._random_p_seq(n)
        ss_partition_fn = get_ss_partition_fn(em)
        vien = ss_partition_fn(p_seq)
        brute = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, matching_to_db(match), em))
        print(n, brute, vien)
        self.assertAlmostEqual(brute, vien, places=7)

    def _max_ss_test(self, em, n):
        p_seq = self._random_p_seq(n)
        ss_partition_fn = get_ss_partition_fn(em, n)
        vien = ss_partition_fn(p_seq)
        max = reference.ss_partition(p_seq, em)
        print(n, max, vien)
        self.assertAlmostEqual(max, vien, places=7)

    def _random_test(self, n):
        em = energy.RandomModel()
        # em = energy.RandomMultiloopModel()
        # em = energy.RandomBulgeModel()
        # em = energy.RandomILModel()
        # em = energy.RandomHairpinModel()

        self._brute_model_test(em, n)
        # self._max_ss_test(em, n)


    def _test_all_1_model_to_10(self):
        print("Starting test: [test_all_1_model_to_10]")
        for n in range(1, 10):
            self._all_1_test(n)

    def _test_all_1_model_12(self):
        # First multiloop for HAIRPIN=3
        print("Starting test: [test_all_1_model_12]")
        self._all_1_test(12)

    def _test_all_1_model_20(self):
        print("Starting test: [test_all_1_model_20]")
        self._all_1_test(20)


    def _test_random_model_to_10(self):
        print("Starting test: [test_random_model_to_10]")
        for n in range(1, 10):
            self._random_test(n)

    def test_nn_to_8(self):

        em = energy.JaxNNModel()

        n_seq = 3
        # for n in range(10, 20): # next do 29, then 30, then 31, 32
        for n in range(1, 8):
            ss_partition_fn = get_ss_partition_fn(em, n)

            for _ in range(n_seq):
                seq = get_rand_seq(n)
                p_seq = jnp.array(seq_to_one_hot(seq))

                start = time.time()
                pf = jit(ss_partition_fn)(p_seq)
                end = time.time()
                print(f"Fuzzy-compatible SS partition took: {onp.round(end - start, 2)} seconds")

                start = time.time()
                discrete_pf = dp_discrete.compute_pf(seq)
                end = time.time()
                print(f"Discrete-compatible SS partition took: {onp.round(end - start, 2)} seconds")
                print(seq, discrete_pf, pf, onp.abs(discrete_pf - pf))

                self.assertAlmostEqual(discrete_pf, pf, places=7)


                """
                em = energy.NNModel()
                start = time.time()
                max_pf = reference.ss_partition(p_seq, em)
                end = time.time()
                print(f"Non-vmapped fuzzy-compatible SS partition took: {onp.round(end - start, 2)} seconds")

                # print(seq, discrete_pf, pf, max_pf)

                self.assertAlmostEqual(max_pf, pf, places=7)
                """



def train(seq_logits, em, n_iter, print_every):

    ss_partition_fn = get_ss_partition_fn(em)
    ss_partition_fn = jit(ss_partition_fn)
    # ss_partition_grad = value_and_grad(ss_partition_fn)
    # ss_partition_grad = jit(ss_partition_grad)

    # Define a lambda that converts logits to p_seq in the desired way
    # FIXME: could have an option for gumbal
    def fold_logits(params):
        curr_logits = params['seq_logits']
        p_seq = jax.nn.softmax(curr_logits)
        return ss_partition_fn(p_seq)
    fold_logits_grad = value_and_grad(fold_logits)
    fold_logits_grad = jit(fold_logits_grad)

    # Note that we use an optimizer from optax: https://github.com/deepmind/optax
    # learning_rate = 1e-1
    learning_rate = 1e-1
    optimizer = optax.adam(learning_rate)
    params = {'seq_logits': seq_logits}
    opt_state = optimizer.init(params)

    all_times = list()
    for i in range(n_iter):
        start = time.time()

        loss, _grad = fold_logits_grad(params)
        updates, opt_state = optimizer.update(_grad, opt_state)
        params = optax.apply_updates(params, updates)

        end = time.time()
        iter_time = end - start
        all_times.append(end - start)

        if i % print_every == 0:
            print(f"{i}: {loss}")

    return params, all_times

def test_train(n, n_iter, print_every=10):
    lo = 0
    hi = 100

    seq_logits = onp.random.uniform(low=lo, high=hi, size=(n, 4))
    seq_logits = jnp.array(seq_logits, dtype=jnp.float64)

    em = energy.JaxNNModel()

    return train(seq_logits, em, n_iter, print_every)


if __name__ == "__main__":

    unittest.main()
    pdb.set_trace()


    # final_params, all_times = test_train(n=46, n_iter=5, print_every=1)
    # pdb.set_trace()

    # NN testing
    """
    seq = "CAAAG"
    # seq = "GUUGC"
    p_seq = jnp.array(seq_to_one_hot(seq))

    start = time.time()
    discrete_pf = dp_discrete.compute_pf(seq)
    end = time.time()
    print(f"Discrete-compatible SS partition took: {onp.round(end - start, 2)} seconds")

    em = energy.JaxNNModel()
    ss_partition_fn = get_ss_partition_fn(em)

    start = time.time()
    pf = ss_partition_fn(p_seq)
    end = time.time()
    print(f"Fuzzy-compatible SS partition took: {onp.round(end - start, 2)} seconds")

    print(seq, discrete_pf, pf)
    pdb.set_trace()
    """


    from d2 import reference


    n = 256
    # n = 16

    if n >= MAX_PRECOMPUTE:
        pdb.set_trace()

    p_seq = onp.empty((n, 4), dtype=onp.float64)
    for i in range(n):
        p_seq[i] = onp.random.random_sample(4)
        p_seq[i] /= onp.sum(p_seq[i])
    p_seq = jnp.array(p_seq)


    em = energy.JaxNNModel()
    ss_partition_fn = get_ss_partition_fn(em, n)
    grad_ss_partition_fn = jit(grad(ss_partition_fn))

    start = time.time()
    # ss_partition_fn(p_seq)
    # our_val, our_grad = jit(value_and_grad(ss_partition_fn))(p_seq)
    our_grad = grad_ss_partition_fn(p_seq)
    # our_val = jit(ss_partition_fn)(p_seq)
    # reference_val = reference.ss_partition(p_seq, em)
    end = time.time()
    print(f"Total time: {onp.round(end - start, 2)}")

    start = time.time()
    our_grad = grad_ss_partition_fn(p_seq)
    end = time.time()
    print(f"Total time: {onp.round(end - start, 2)}")
    pdb.set_trace()
