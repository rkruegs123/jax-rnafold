import pdb
import functools
import math
from typing import Callable

import jax
from jax import vmap, jit, lax
import jax.numpy as jnp

from jax_rnafold.d0 import energy
from jax_rnafold.common.checkpoint import checkpoint_scan
from jax_rnafold.common.utils import bp_bases, N4, NBPS
from jax_rnafold.common.utils import MAX_PRECOMPUTE, MAX_LOOP

jax.config.update("jax_enable_x64", True)



f64 = jnp.float64

def get_ss_partition_fn(em: energy.Model, seq_len: int, max_loop: int = MAX_LOOP,
                        scale: float = 0.0, checkpoint_every: int = 10) -> Callable:
    """
    Constructs a JIT-compatible function for computing the structure-sequence
    partition function via the generalized McCaskill's algorithm. The returned
    function takes as input a probabilistic sequence of shape `[seq_len, 4]`.

    Args:
      em: An energy model.
      seq_len: The length of the input sequence.
      max_loop: The maximum loop size for internal loops and bulge loops. Defaults to 30.
      scale: The exponent for Boltzmann rescaling. The recommended value is `-3/2 * seq_len`. A scale of 0.0 corresponds to no rescaling (as :math:`e^0 = 1`).
      checkpoint_every: The frequency of checkpointing for gradient rematerialization.

    Returns:
      A function that accepts a probabilistic sequence of shape `[seq_len, 4]` as input and returns the computed structure-sequence partition function.
    """

    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint_scan,
                                 checkpoint_every=checkpoint_every)

    two_loop_length = min(seq_len, max_loop)

    s_table = jnp.array([math.exp(i*scale/seq_len) for i in range(seq_len+5)])

    special_hairpin_lens = em.special_hairpin_lens
    max_sp_hairpin_len_up = special_hairpin_lens.max() - 2 # Subtract 2 for the paired nt
    special_hairpin_idxs = em.special_hairpin_idxs
    special_hairpin_start_pos = em.special_hairpin_start_pos
    n_special_hairpins = em.n_special_hairpins


    @jit
    def fill_external(i, E, P, padded_p_seq):
        def get_j_bp_term(j, bp_idx):
            cond = (j >= i+1) & (j < seq_len)

            bp = bp_bases[bp_idx]
            bi = bp[0]
            bj = bp[1]
            base_en = E[j+1]*padded_p_seq[i, bi]*padded_p_seq[j, bj]
            return jnp.where(cond, base_en*P[bp_idx, i, j]*em.en_ext_branch(bi, bj)*s_table[2], 0.0)
        get_all_terms = vmap(vmap(get_j_bp_term, (0, None)), (None, 0))

        sm = E[i+1]*s_table[1] + jnp.sum(get_all_terms(jnp.arange(seq_len), jnp.arange(NBPS)))
        E = E.at[i].set(sm)
        return E


    @jit
    def fill_outer_mismatch(i, OMM, P, padded_p_seq):

        def compute_term(j, bp_idx, bim1, bjp1):
            bp = bp_bases[bp_idx]
            bi = bp[0]
            bj = bp[1]
            return em.en_il_outer_mismatch(bi, bj, bim1, bjp1) \
                * P[bp_idx, i, j] * padded_p_seq[i-1, bim1] * padded_p_seq[j+1, bjp1] \
                * padded_p_seq[i, bi] * padded_p_seq[j, bj]
        compute_all_terms = vmap(vmap(vmap(compute_term,
                                           (None, 0, None, None)),
                                      (None, None, 0, None)),
                                 (None, None, None, 0))

        @jit
        def get_j_sm(j):
            cond = (j >= i) & (j < seq_len)
            j_sm = jnp.sum(compute_all_terms(j, jnp.arange(NBPS), N4, N4))
            return jnp.where(cond, j_sm, 0.0)

        all_js = jnp.arange(seq_len+1)
        all_j_sms = vmap(get_j_sm)(all_js)
        OMM = OMM.at[i].add(all_j_sms)
        return OMM


    @jit
    def fill_multibranch(i, MB, P, padded_p_seq):

        def compute_term(j, bp_idx):
            bp = bp_bases[bp_idx]
            bi = bp[0]
            bj = bp[1]
            return P[bp_idx, i, j] * em.en_multi_branch(bi, bj) \
                * padded_p_seq[i, bi] * padded_p_seq[j, bj] * s_table[2]
        compute_all_terms = vmap(compute_term, (None, 0))

        @jit
        def get_j_sm(j):
            cond = (j >= i) & (j < seq_len+1)
            j_sm = jnp.sum(compute_all_terms(j, jnp.arange(NBPS)))
            return jnp.where(cond, j_sm, 0.0)

        all_js = jnp.arange(seq_len+1)
        all_j_sms = vmap(get_j_sm)(all_js)
        MB = MB.at[i].add(all_j_sms)
        return MB


    @jit
    def pr_special_hairpin(id, i, j, padded_p_seq):
        start_pos = special_hairpin_start_pos[id]
        id_len = special_hairpin_lens[id]
        def get_sp_hairpin_nuc_prob(k_offset):
            k = i + 1 + k_offset
            cond = (k >= i+1) & (k < j)
            idx_pos = start_pos + 1 + k_offset
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
            correction = pr_special_hairpin(id, i, j, padded_p_seq) \
                         * em.en_hairpin_not_special(bi, bj, bip1, bjm1, sp_hairpin_len - 2)
            return jnp.where(id_valid, correction, 0.0)

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

            val = pr_special_hairpin(id, i, j, padded_p_seq) * em.en_hairpin_special(id)
            return jnp.where(id_valid, val, 0.0)

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

        def get_bp_kl(bp_idx, kl_offset):
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]
            bp_kl_sm = 0


            # Right bulge
            l = j-2-kl_offset
            right_cond = (l >= i+2)
            right_val = P[bp_idx, i+1, l]*padded_p_seq[i+1, bk] * \
                padded_p_seq[l, bl]*em.en_bulge(bi, bj, bk, bl, j-l-1) * \
                s_table[j-l+1]
            bp_kl_sm += jnp.where(right_cond, right_val, 0.0)

            # Left bulge
            k = i+2+kl_offset
            left_cond = (k < j-1)
            left_val = P[bp_idx, k, j-1]*padded_p_seq[k, bk] * \
                padded_p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, k-i-1) * \
                s_table[k-i+1]
            bp_kl_sm += jnp.where(left_cond, left_val, 0.0)

            return bp_kl_sm

        def get_bp_all_kl(bp_idx):
            all_kl_offsets = jnp.arange(two_loop_length)
            all_bp_kl_sms = vmap(get_bp_kl, (None, 0))(bp_idx, all_kl_offsets)
            return jnp.sum(all_bp_kl_sms)

        all_bp_sms = vmap(get_bp_all_kl)(jnp.arange(NBPS))
        return jnp.sum(all_bp_sms)


    @jit
    def psum_internal_loops(bi, bj, i, j, padded_p_seq, P, OMM):
        def get_mmij_term(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
        mmij_terms = vmap(vmap(get_mmij_term, (0, None)), (None, 0))(N4, N4)
        mmij = jnp.sum(mmij_terms)

        sm = 0.0

        # Note: 1x1 and 1xN and Nx1. Not just 1xN.
        @jit
        def get_bp_1n_sm(bp_idx, bip1, bjm1):
            bp_1n_sm = 0.0
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]

            pr_ij_mm = padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1]
            # 1x1. Don't need safe_P since we pad on both sides.
            bp_1n_sm += P[bp_idx, i+2, j-2]*padded_p_seq[i+2, bk] \
                        * padded_p_seq[j-2, bl]*pr_ij_mm \
                        * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bip1, bjm1, 1, 1) \
                        * s_table[4]

            # FIXME: change to z_offset or kl_offset
            def z_b_fn(z_offset, b):
                z_b_sm = 0.0

                l = j-3-z_offset
                l_cond = (l >= i+3)
                il_en = em.en_internal(
                     bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-l-1)
                right_term = P[bp_idx, i+2, l]*padded_p_seq[i+2, bk] \
                             * padded_p_seq[l, bl]*padded_p_seq[l+1, b]*pr_ij_mm*il_en \
                             * s_table[j-l+2]
                z_b_sm += jnp.where(l_cond, right_term, 0.0)

                k = i+3+z_offset
                k_cond = (k < j-2)
                il_en = em.en_internal(
                     bi, bj, bk, bl, bip1, bjm1, b, bjm1, k-i-1, 1)
                left_term = P[bp_idx, k, j-2]*padded_p_seq[k, bk] \
                           * padded_p_seq[j-2, bl]*padded_p_seq[k-1, b]*pr_ij_mm*il_en \
                           * s_table[k-i+2]
                z_b_sm += jnp.where(k_cond, left_term, 0.0)

                return z_b_sm

            get_all_zb_terms = vmap(vmap(z_b_fn, (0, None)), (None, 0))

            # z_offsets = jnp.arange(seq_len+1)
            z_offsets = jnp.arange(two_loop_length)
            bp_1n_sm += jnp.sum(get_all_zb_terms(z_offsets, N4))
            return bp_1n_sm
        get_all_1n_terms = vmap(vmap(vmap(get_bp_1n_sm, (0, None, None)),
                                     (None, 0, None)), (None, None, 0))
        sm += jnp.sum(get_all_1n_terms(jnp.arange(NBPS), N4, N4))


        # 2x2, 3x2, 2x3
        def get_bp_22_23_32_sm(bp_idx, k_offset, l_offset):
            k = i + k_offset + 2
            l = j - l_offset - 2

            bp = bp_bases[bp_idx]
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
                return P[bp_idx, k, l]*padded_p_seq[k, bk]*padded_p_seq[l, bl] \
                    * em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) \
                    * padded_p_seq[k-1, bkm1]*padded_p_seq[l+1, blp1] \
                    * padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] \
                    * s_table[lup+rup+2]
            get_all_summands = vmap(get_bp_22_23_32_summand, (None, None, None, 0))
            get_all_summands = vmap(get_all_summands, (None, None, 0, None))
            get_all_summands = vmap(get_all_summands, (None, 0, None, None))
            get_all_summands = vmap(get_all_summands, (0, None, None, None))

            all_summands = get_all_summands(N4, N4, N4, N4)
            return jnp.where(cond, jnp.sum(all_summands), 0.0)
        get_all_special_terms = vmap(vmap(vmap(get_bp_22_23_32_sm, (0, None, None)),
                                          (None, 0, None)), (None, None, 0))
        sm += jnp.sum(get_all_special_terms(jnp.arange(NBPS), jnp.arange(3), jnp.arange(3)))


        # general internal loops
        def general_kl_sm(k_offset, l_offset):
            k = k_offset + i + 2
            l = j - l_offset - 2

            lup = k-i-1
            rup = j-l-1

            # idx_cond = (k >= i+2) & (k < j-2) & (l >= k+1) & (l < j-1)
            idx_cond = (k < l)
            is_not_n1 = (lup > 1) & (rup > 1)
            is_22_23_32 = ((lup == 2) & (rup == 2)) \
                          | ((lup == 2) & (rup == 3)) \
                          | ((lup == 3) & (rup == 2))
            cond = idx_cond & is_not_n1 & ~is_22_23_32

            general_term = em.en_internal_init(lup+rup) * em.en_internal_asym(lup, rup) \
                           * OMM[k, l] * mmij * s_table[lup+rup+2]

            return jnp.where(cond, general_term, 0.0)
        get_all_general = vmap(vmap(general_kl_sm, (0, None)), (None, 0))
        # k_offsets, l_offsets = jnp.arange(seq_len+1), jnp.arange(seq_len+1)
        k_offsets, l_offsets = jnp.arange(two_loop_length), jnp.arange(two_loop_length)
        sm += jnp.sum(get_all_general(k_offsets, l_offsets))

        return sm

    @jit
    def fill_paired(i, padded_p_seq, OMM, ML, P):

        def get_bp_stack(bp_idx, j, bi, bj):
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]
            return P[bp_idx, i+1, j-1]*padded_p_seq[i+1, bk] * \
                padded_p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)

        # For a given bp and j, get the corresponding sum
        def get_bp_j_sm(bp_idx, j):
            bp = bp_bases[bp_idx]
            bi = bp[0]
            bj = bp[1]
            sm = psum_hairpin(bi, bj, i, j, padded_p_seq) * s_table[j-i-1]
            sm += psum_bulges(bi, bj, i, j, padded_p_seq, P)
            sm += psum_internal_loops(bi, bj, i, j, padded_p_seq, P, OMM)

            # Stacks
            stack_summands = vmap(get_bp_stack, (0, None, None, None))(jnp.arange(NBPS), j, bi, bj)
            sm += jnp.sum(stack_summands) * s_table[2]

            # Multi-loops
            sm += em.en_multi_closing(bi, bj)*ML[2, i+1, j-1]

            cond = (j >= i+em.hairpin+1) & (j < seq_len)
            return jnp.where(cond, sm, P[bp_idx, i, j])

        # For a fixed base pair, get all js
        def get_bp_all_js(bp_idx):
            js = jnp.arange(seq_len+1)
            return vmap(get_bp_j_sm, (None, 0))(bp_idx, js)

        all_bp_js = vmap(get_bp_all_js)(jnp.arange(NBPS))
        P = P.at[:, i].set(all_bp_js)
        return P

    def fill_multi(i, padded_p_seq, ML, MB):
        def nb_j_fn(nb, j):
            nb_j_cond = (j >= i) & (j < seq_len)
            nb_j_sm = ML[nb, i+1, j] * s_table[1]

            idx = jnp.where(nb-1 > 0, nb-1, 0)
            def k_fn(k):
                k_cond = (k >= i) & (k < j+1)
                return jnp.where(k_cond, ML[idx, k+1, j] * MB[i, k], 0.0)
            nb_j_sm += jnp.sum(vmap(k_fn)(jnp.arange(seq_len+1)))

            return jnp.where(nb_j_cond, nb_j_sm, ML[nb, i, j])
        get_nb_j_terms = vmap(vmap(nb_j_fn, (None, 0)), (0, None))

        nb_j_terms = get_nb_j_terms(jnp.arange(3), jnp.arange(seq_len+1))
        ML = ML.at[:, i, :].set(nb_j_terms)
        return ML


    def ss_partition(p_seq):

        # Pad appropriately
        padded_p_seq = jnp.zeros((seq_len+1, 4), dtype=f64)
        padded_p_seq = padded_p_seq.at[:seq_len].set(p_seq)

        E = jnp.zeros((seq_len+1), dtype=f64)
        P = jnp.zeros((NBPS, seq_len+1, seq_len+1), dtype=f64)
        ML = jnp.zeros((3, seq_len+1, seq_len+1), dtype=f64)
        MB = jnp.zeros((seq_len+1, seq_len+1), dtype=f64)
        OMM = jnp.zeros((seq_len+1, seq_len+1), dtype=f64)
        E = E.at[seq_len].set(1)
        ML = ML.at[0, :, :].set(1)

        @jit
        def fill_table(carry, i):
            OMM, P, ML, MB, E = carry

            P = fill_paired(i, padded_p_seq, OMM, ML, P)
            OMM = fill_outer_mismatch(i, OMM, P, padded_p_seq)
            MB = fill_multibranch(i, MB, P, padded_p_seq)
            ML = fill_multi(i, padded_p_seq, ML, MB)
            E = fill_external(i, E, P, padded_p_seq)

            return (OMM, P, ML, MB, E), None

        (OMM, P, ML, MB, E), _ = scan(fill_table,
                                      (OMM, P, ML, MB, E),
                                      jnp.arange(seq_len-1, -1, -1))

        return E[0]

    return ss_partition
