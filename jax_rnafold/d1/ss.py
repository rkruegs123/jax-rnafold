import numpy as onp
import pdb
import functools
import unittest
import time
from tqdm import tqdm

import jax
import optax
from jax import vmap, jit, grad, value_and_grad
from jax.tree_util import Partial
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from jax_rnafold.d1 import energy
from jax_rnafold.common.checkpoint import checkpoint_scan
from jax_rnafold.common.utils import bp_bases, HAIRPIN, N4, INVALID_BASE
from jax_rnafold.common.utils import matching_to_db
from jax_rnafold.common.utils import MAX_PRECOMPUTE, MAX_LOOP
from jax_rnafold.common import brute_force
from jax_rnafold.common import nussinov as nus
from jax_rnafold.common.utils import get_rand_seq, seq_to_one_hot, random_pseq, matching_2_dot_bracket, bcolors



f64 = jnp.float64


checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)



def get_ss_partition_fn(em : energy.NNModel, seq_len, max_loop=MAX_LOOP):
    two_loop_length = min(seq_len, max_loop)

    special_hairpin_lens = em.nn_params.special_hairpin_lens
    special_hairpin_idxs = em.nn_params.special_hairpin_idxs
    special_hairpin_start_pos = em.nn_params.special_hairpin_start_pos
    n_special_hairpins = em.nn_params.n_special_hairpins

    @jit
    def fill_outer_mismatch(k, OMM, padded_p_seq):

        # For a fixed baes pair and mismatch, get the energy
        def get_bp_mm(bk, bl, bkm1, blp1, k, l):
            return em.en_il_outer_mismatch(bk, bl, bkm1, blp1,
                                           em.nn_params.params['mismatch_interior']) \
                * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1]

        # For a single l, get 6 values corresponding to the sum over each mismatch for each base pair
        def get_all_bp_all_mms(l):
            cond = (l >= k+1) & (l < seq_len)

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
                return jnp.sum(all_mms)

            true_val = vmap(get_bp_all_mms)(bp_bases)
            return jnp.where(cond, true_val, jnp.zeros(6)) # Note: set to 0 as we eventually *add* to OMM rather than *set*

        ls = jnp.arange(seq_len+2)
        all_bp_mms = vmap(get_all_bp_all_mms)(ls)
        # OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T) # Note how we take the transpose here.
        OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T)
        return OMM

    def fill_multi(i, padded_p_seq, ML, P):
        def nb_j_fn(nb, j):
            j_cond = (j > i) & (j < seq_len+1)
            b_idx = jnp.where(nb-1 > 0, nb-1, 0)

            def k_fn(k):
                k_cond = (k > i) & (k < j+1)

                def bi_bk_fn(bi, bk):

                    base_en = ML[b_idx, k+1, j] * padded_p_seq[i, bi] \
                              * padded_p_seq[k, bk]
                    k_bi_bk_sm = base_en * P[bi, bk, i, k] * em.en_multi_branch(bi, bk)

                    bip1_fn = lambda bip1: base_en*P[bip1, bk, i+1, k] * em.en_multi_branch(bip1, bk) \
                              * padded_p_seq[i+1, bip1] * em.en_5dangle(bi, bip1, bk)
                    bip1_sm = jnp.sum(vmap(bip1_fn)(N4))

                    bkm1_fn = lambda bkm1: base_en * P[bi, bkm1, i, k-1] \
                              * em.en_multi_branch(bi, bkm1) * padded_p_seq[k-1, bkm1] \
                              * em.en_3dangle(bi, bkm1, bk)
                    bkm1_sm = jnp.sum(vmap(bkm1_fn)(N4))

                    bip1_bkm1_fn = lambda bip1, bkm1: base_en * P[bip1, bkm1, i+1, k-1] \
                                   * em.en_multi_branch(bip1, bkm1) * padded_p_seq[i+1, bip1] \
                                   * padded_p_seq[k-1, bkm1] * em.en_term_mismatch(bi, bip1, bkm1, bk)
                    bip1_bkm1_sm = vmap(vmap(bip1_bkm1_fn, (0, None)), (None, 0))(N4, N4)

                    return k_bi_bk_sm + bip1_sm + bkm1_sm + bip1_bkm1_sm

                bi_bk_sm = jnp.sum(vmap(vmap(bi_bk_fn, (0, None)), (None, 0))(N4, N4))
                return jnp.where(k_cond, bi_bk_sm, 0.0)

            k_sm = jnp.sum(vmap(k_fn)(jnp.arange(seq_len+2)))
            return jnp.where(j_cond, k_sm + ML[nb, i+1, j], ML[nb, i, j])

        get_all_nb_j = vmap(vmap(nb_j_fn, (None, 0)), (0, None))
        all_nb_j = get_all_nb_j(jnp.arange(3), jnp.arange(seq_len + 2))

        return ML.at[:, i, :].set(all_nb_j)


    def fill_external(i, padded_p_seq, P, E):
        def j_fn(j):
            cond = (j >= i+1) & (j < seq_len+1)

            def bi_bj_fn(bi, bj):
                base_en = E[j+1]*padded_p_seq[i, bi]*padded_p_seq[j, bj]
                bi_bj_sm = base_en * P[bi, bj, i, j] * em.en_ext_branch(bi, bj)

                bip1_fn = lambda bip1: base_en * P[bip1, bj, i+1, j] \
                          * em.en_ext_branch(bip1, bj) * padded_p_seq[i+1, bip1] \
                          * em.en_5dangle(bi, bip1, bj)
                bip1_sm = jnp.sum(vmap(bip1_fn)(N4))

                bjm1_fn = lambda bjm1: base_en * P[bi, bjm1, i, j-1] \
                          * em.en_ext_branch(bi, bjm1) * padded_p_seq[j-1, bjm1] \
                          * em.en_3dangle(bi, bjm1, bj)
                bjm1_sm = jnp.sum(vmap(bjm1_fn)(N4))

                bip1_bjm1_fn = lambda bip1, bjm1: base_en * P[bip1, bjm1, i+1, j-1] \
                               * em.en_ext_branch(bip1, bjm1) * padded_p_seq[j-1, bjm1] \
                               * padded_p_seq[i+1, bip1] * em.en_term_mismatch(bi, bip1, bjm1, bj)
                bip1_bjm1_sm = jnp.sum(vmap(vmap(bip1_bjm1_fn, (0, None)), (None, 0))(N4, N4))

                bi_bj_sm += bip1_sm + bjm1_sm + bip1_bjm1_sm
                return bi_bj_sm

            j_sm = jnp.sum(vmap(vmap(bi_bj_fn, (0, None)), (None, 0))(N4, N4))
            return jnp.where(cond, j_sm, 0.0)

        all_j_sm = jnp.sum(vmap(j_fn)(jnp.arange(seq_len+1)))
        sm = all_j_sm + E[i+1]

        return E.at[i].set(sm)



    @jit
    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1
        u = j - i - 1

        # FIXME: repeated computation should combine with `psum_hairpin_special()`
        def pr_special_hairpin(id, i, j):
            start_pos = special_hairpin_start_pos[id]
            id_len = special_hairpin_lens[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, special_hairpin_idxs[idx_pos]], 1.0)
            ks = jnp.arange(seq_len+1) # FIXME: is this correct?
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        def special_hairpin_correction(id):
            sp_hairpin_len = special_hairpin_lens[id]
            start_pos = special_hairpin_start_pos[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (special_hairpin_lens[id] == up2) \
                       & (special_hairpin_idxs[start_pos] == bi) \
                       & (special_hairpin_idxs[end_pos] == bj)

            bjm1 = special_hairpin_idxs[end_pos - 1]
            bip1 = special_hairpin_idxs[start_pos + 1]
            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_not_special(
                                 bi, bj, bip1, bjm1, sp_hairpin_len - 2),
                             0.0)

        summands = vmap(special_hairpin_correction)(jnp.arange(n_special_hairpins))
        sm = jnp.sum(summands)
        return sm

    @jit
    def psum_hairpin_special(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1

        def pr_special_hairpin(id, i, j):
            start_pos = special_hairpin_start_pos[id]
            id_len = special_hairpin_lens[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, special_hairpin_idxs[idx_pos]], 1.0)
            ks = jnp.arange(seq_len+1) # FIXME: is this correct?
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        def special_hairpin(id):
            sp_hairpin_len = special_hairpin_lens[id]
            start_pos = special_hairpin_start_pos[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (special_hairpin_lens[id] == up2) \
                       & (special_hairpin_idxs[start_pos] == bi) \
                       & (special_hairpin_idxs[end_pos] == bj)

            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_special(id),
                             0.0)

        summands = vmap(special_hairpin)(jnp.arange(n_special_hairpins))
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
                em.en_il_inner_mismatch(bi, bj, bip1, bjm1,
                                        mm_table=em.nn_params.params['mismatch_interior'])
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
    def psum_multiloops(bi, bj, i, j, padded_p_seq, ML):
        closing_en = em.en_multi_closing(bi, bj)
        sm = closing_en*ML[2, i+1, j-1]

        bip1_fn = lambda bip1: closing_en * ML[2, i+2, j-1] \
                  * padded_p_seq[i+1, bip1] * em.en_3dangle_inner(bi, bip1, bj)
        bip1_sm = jnp.sum(vmap(bip1_fn)(N4))

        bjm1_fn = lambda bjm1: closing_en * ML[2, i+1, j-2] \
                  * padded_p_seq[j-1, bjm1] * em.en_5dangle_inner(bi, bjm1, bj)
        bjm1_sm = jnp.sum(vmap(bjm1_fn)(N4))

        bip1_bjm1_fn = lambda bip1, bjm1: closing_en * ML[2, i+2, j-2] \
                       * padded_p_seq[i+1, bip1] * padded_p_seq[j-1, bjm1] \
                       * em.en_term_mismatch_inner(bi, bip1, bjm1, bj)
        bip1_bjm1_sm = jnp.sum(vmap(vmap(bip1_bjm1_fn, (None, 0)), (0, None))(N4, N4))

        sm += bip1_sm + bjm1_sm + bip1_bjm1_sm
        return sm

    @jit
    def fill_paired(i, padded_p_seq, OMM, ML, P):

        def get_bp_stack(bp, j, bi, bj):
            bk = bp[0]
            bl = bp[1]
            return P[bk, bl, i+1, j-1]*padded_p_seq[i+1, bk] * \
                padded_p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)


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
            sm += psum_multiloops(bi, bj, i, j, padded_p_seq, ML)

            cond = (j >= i+HAIRPIN+1) & (j < seq_len+1)
            return jnp.where(cond, sm, P[bi, bj, i, j])


        # For a fixed base pair, get all js
        def get_bp_all_js(bp):
            js = jnp.arange(seq_len+2)
            return vmap(get_bp_j_sm, (None, 0))(bp, js)

        all_bp_js = vmap(get_bp_all_js)(bp_bases)
        P = P.at[bp_bases[:, 0], bp_bases[:, 1], i].set(all_bp_js) # Note how we *dont* take the transpose here
        return P

    def ss_partition(p_seq):

        # Pad appropriately
        padded_p_seq = jnp.zeros((seq_len+2, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[1:seq_len+1].set(p_seq)
        padded_p_seq = padded_p_seq.at[0, 0].set(1.0)
        padded_p_seq = padded_p_seq.at[-1, 0].set(1.0)

        E = jnp.zeros((seq_len+2), dtype=f64)
        P = jnp.zeros((4, 4, seq_len+2, seq_len+2), dtype=f64)
        ML = jnp.zeros((3, seq_len+2, seq_len+2), dtype=f64)
        OMM = jnp.zeros((4, 4, seq_len+2, seq_len+2), dtype=f64)
        E = E.at[seq_len+1].set(1)
        ML = ML.at[0, :, :].set(1)

        @jit
        def fill_table(carry, i):
            OMM, P, ML, E = carry

            OMM = fill_outer_mismatch(i, OMM, padded_p_seq)
            P = fill_paired(i, padded_p_seq, OMM, ML, P)
            # ML = fill_multi(i, padded_p_seq, ML, P)
            E = fill_external(i, padded_p_seq, P, E)

            return (OMM, P, ML, E), None

        (OMM, P, ML, E), _ = scan(fill_table,
                                  (OMM, P, ML, E),
                                  jnp.arange(seq_len, 0, -1))

        return E[1]

    return ss_partition


def train(n, lr=0.1, n_iter=10, print_every=1):
    em = energy.JaxNNModel()
    ss_fn = get_ss_partition_fn(em, n)

    def loss_fn(params):
        curr_logits = params['seq_logits']
        p_seq = jax.nn.softmax(curr_logits)
        q = ss_fn(p_seq)
        return q
    grad_fn = value_and_grad(loss_fn)
    grad_fn = jit(grad_fn)

    seq_logits = onp.full((n, 4), 5)
    seq_logits = jnp.array(seq_logits, dtype=jnp.float64)
    params = {'seq_logits': seq_logits}

    optimizer = optax.rmsprop(learning_rate=lr)
    opt_state = optimizer.init(params)

    for i in tqdm(range(n_iter)):
        start = time.time()
        q, _grad = grad_fn(params)

        updates, opt_state = optimizer.update(_grad, opt_state)
        params = optax.apply_updates(params, updates)

        end = time.time()
        iter_time = end - start

        if i % print_every == 0:
            print(f"Iteration {i}:")
            print(f"- Q: {q}")
            print(f"- Time: {onp.round(iter_time, 2)}")

    return q



class TestSSPartitionFunction(unittest.TestCase):


    def _random_seq_test(self, n, em):
        import jax_rnafold.d1.ss_reference as reference

        p_seq = random_pseq(n)

        ss_fn = get_ss_partition_fn(em, n)
        q = ss_fn(p_seq)

        # q_ref = reference.ss_partition(p_seq, em)

        brute_q = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, matching_to_db(match), em))
        print(n, brute_q, q)
        self.assertAlmostEqual(brute_q, q, places=7)

    def _test_all_1_model_to_10(self):
        for n in range(1, 10):
            self._random_seq_test(n, energy.All1Model())

    def _test_random_model_to_10(self):
        for n in range(1, 10):
            self._random_seq_test(n, energy.RandomModel())

    def fuzz_test(self, n, num_seq, em, tol_places=6, max_structs=20):
        import random

        from jax_rnafold.common import vienna_rna
        import jax_rnafold.d1.ss_reference as reference


        ss_partition_fn = get_ss_partition_fn(em, n)
        ss_partition_fn = jit(ss_partition_fn)

        seqs = [get_rand_seq(n) for _ in range(num_seq)]

        failed_cases = list()
        n_passed = 0

        failed_cases = list()
        for seq in seqs:
            p_seq = jnp.array(seq_to_one_hot(seq))

            print(f"Sequence: {seq}")

            ss_pf = ss_partition_fn(p_seq)
            print(f"\tComputed partition function: {ss_pf}")
            """
            ref_pf = vienna_rna.get_vienna_pf(seq, dangle_mode=0)
            print(f"\tVienna partition function: {ref_pf}")
            ref_pf = brute_force.ss_partition(p_seq,
                                              energy_fn=lambda seq, match: energy.calculate(
                                                  seq, matching_to_db(match), em))
            print(f"\tBrute partition function: {ref_pf}")
            """
            ref_pf = reference.ss_partition(p_seq, energy.StandardNNModel())
            print(f"\tReference partition function: {ref_pf}")

            self.assertAlmostEqual(ss_pf, ref_pf, places=7)

    def _test_fuzz(self):
        em = energy.JaxNNModel()
        self.fuzz_test(16, 10, em)

    def test_fuzz(self):
        # Note: failing ACAACAGUAACUCCUA, GUUUAUGGU
        em = energy.JaxNNModel()
        self.fuzz_test(n=9, num_seq=10, em=em)

    def _test_train(self):
        hi = train(n=64)
        self.assertAlmostEqual(1.0, hi, places=7)


if __name__ == "__main__":
    unittest.main()
    # train(8)
