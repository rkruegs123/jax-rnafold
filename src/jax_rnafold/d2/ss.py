import numpy as onp
import pdb
import functools
import time

import jax
from jax import vmap, jit
from jax.tree_util import Partial
import jax.numpy as jnp

from jax_rnafold.common.checkpoint import checkpoint_scan
from jax_rnafold.common.utils import bp_bases, N4, INVALID_BASE
from jax_rnafold.common.utils import MAX_PRECOMPUTE, MAX_LOOP
from jax_rnafold.d2 import energy

jax.config.update("jax_enable_x64", True)


f64 = jnp.float64


# TODO: remove all unnecessary n parameterizations
def get_ss_partition_fn(em : energy.Model, seq_len, max_loop=MAX_LOOP, checkpoint_every=None):

    if checkpoint_every is None:
        scan = jax.lax.scan
    else:
        scan = functools.partial(checkpoint_scan,
                                 checkpoint_every=checkpoint_every)

    two_loop_length = min(seq_len, max_loop)

    special_hairpin_lens = em.special_hairpin_lens
    max_sp_hairpin_len_up = special_hairpin_lens.max() - 2 # Subtract 2 for the paired nt
    special_hairpin_idxs = em.special_hairpin_idxs
    special_hairpin_start_pos = em.special_hairpin_start_pos
    special_hairpin_energies = em.special_hairpin_energies
    n_special_hairpins = em.n_special_hairpins

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
                all_mms = mapped_fn(bk, bl, N4, N4, k, l)
                return jnp.sum(all_mms)

            true_val = vmap(get_bp_all_mms)(bp_bases)
            return jnp.where(cond, true_val, jnp.zeros(6)) # Note: set to 0 as we eventually *add* to OMM rather than *set*

        ls = jnp.arange(seq_len+2)
        all_bp_mms = vmap(get_all_bp_all_mms)(ls)
        OMM = OMM.at[bp_bases[:, 0], bp_bases[:, 1], k].add(all_bp_mms.T)
        return OMM


    def pr_special_hairpin(id, i, j, padded_p_seq):
        start_pos = special_hairpin_start_pos[id]
        id_len = special_hairpin_lens[id]
        def get_sp_hairpin_nuc_prob(k_offset):
            k = i + 1 + k_offset
            cond = (k >= i+1) & (k < j)
            idx_pos = start_pos + (k-i)
            return jnp.where(cond, padded_p_seq[k, special_hairpin_idxs[idx_pos]], 1.0)
        k_offsets = jnp.arange(max_sp_hairpin_len_up)
        prs = vmap(get_sp_hairpin_nuc_prob)(k_offsets)
        pr = 1 # we know i and j match
        pr *= jnp.prod(prs)
        return pr

    @jit
    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1
        u = j - i - 1

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
                             pr_special_hairpin(id, i, j, padded_p_seq) * em.en_hairpin_not_special(
                                 bi, bj, bip1, bjm1, sp_hairpin_len - 2),
                             0.0)

        summands = vmap(special_hairpin_correction)(jnp.arange(n_special_hairpins))
        sm = jnp.sum(summands)
        return sm


    @jit
    def psum_hairpin_special(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1

        def special_hairpin(id):
            sp_hairpin_len = special_hairpin_lens[id]
            start_pos = special_hairpin_start_pos[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (special_hairpin_lens[id] == up2) \
                       & (special_hairpin_idxs[start_pos] == bi) \
                       & (special_hairpin_idxs[end_pos] == bj)

            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j, padded_p_seq) * em.en_hairpin_special(id),
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
            bp_kl_sm = 0
            bk = bp[0]
            bl = bp[1]

            # Right bulge
            l = j-2-kl_offset
            right_cond = (l >= i+2)
            right_val = P[bk, bl, i+1, l]*padded_p_seq[i+1, bk] * \
                padded_p_seq[l, bl]*em.en_bulge(bi, bj, bk, bl, j-l-1)
            bp_kl_sm += jnp.where(right_cond, right_val, 0.0)

            # Left bulge
            k = i+2+kl_offset
            left_cond = (k < j-1)
            left_val = P[bk, bl, k, j-1]*padded_p_seq[k, bk] * \
                padded_p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, k-i-1)
            bp_kl_sm += jnp.where(left_cond, left_val, 0.0)

            return bp_kl_sm


        def get_bp_all_kl(bp):
            all_kl_offsets = jnp.arange(two_loop_length)
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

            z_offsets = jnp.arange(two_loop_length)
            all_zs_sms = vmap(get_z_all_bs_sm)(z_offsets)

            bp_1n_sm += jnp.sum(all_zs_sms)
            return bp_1n_sm


        def get_bp_22_23_32_sm(bp, k_offset, l_offset):
            k = i + k_offset + 2
            l = j - l_offset - 2
            bk = bp[0]
            bl = bp[1]
            lup = k-i-1
            rup = j-l-1

            cond_lup_rup = ((lup == 2) & (rup == 2)) \
                | ((lup == 2) & (rup == 3)) \
                | ((lup == 3) & (rup == 2))
            cond_idx = (k < j-2) & (l >= k+1)
            cond = cond_lup_rup & cond_idx

            def get_bp_22_23_32_summand(bip1, bjm1, bkm1, blp1):
                return P[bk, bl, k, l] * padded_p_seq[k, bk] * padded_p_seq[l, bl] \
                    * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) \
                    * padded_p_seq[k-1, bkm1] * padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1] * padded_p_seq[j-1, bjm1]
            get_all_summands = vmap(get_bp_22_23_32_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))
            all_summands = get_all_summands(N4, N4, N4, N4)
            return jnp.where(cond, jnp.sum(all_summands), 0.0)

        def get_bp_general_sm(bp, k_offset, l_offset):
            k = k_offset + i + 2
            l = j - l_offset - 2

            bk = bp[0]
            bl = bp[1]
            idx_cond = (k < l)
            lup = k-i-1
            rup = j-l-1
            is_not_n1 = (lup > 1) & (rup > 1)
            is_22_23_32 = ((lup == 2) & (rup == 2)) \
                          | ((lup == 2) & (rup == 3)) \
                          | ((lup == 3) & (rup == 2))

            cond = idx_cond & is_not_n1 & ~is_22_23_32

            init_and_pair = em.en_internal_init(lup+rup) \
                            * em.en_internal_asym(lup, rup) \
                            * P[bk, bl, k, l] \
                            * padded_p_seq[k, bk]*padded_p_seq[l, bl]
            gen_sm = OMM[bk, bl, k, l]*mmij*init_and_pair

            return jnp.where(cond, gen_sm, 0.0)

        def get_bp_sm(bp):
            bk = bp[0]
            bl = bp[1]
            bp_sum = 0.0

            # Case 1: nx1 and 1xn
            all_1n_sms = vmap(vmap(get_bp_1n_sm, (None, 0, None)), (None, None, 0))(bp, N4, N4)
            bp_sum += jnp.sum(all_1n_sms)

            # Case 2: 2x2, 3x2, and 3x2
            all_22_23_32_sms = vmap(vmap(get_bp_22_23_32_sm, (None, 0, None)), (None, None, 0))(bp, jnp.arange(3), jnp.arange(3))
            bp_sum += jnp.sum(all_22_23_32_sms)

            # Case 3: All others
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

            cond = (j >= i+em.hairpin+1) & (j < seq_len+1)
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
