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

from d1 import energy
from common.checkpoint import checkpoint_scan
from common.utils import bp_bases, HAIRPIN, N4, INVALID_BASE
from common.utils import SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_LENS, \
    SPECIAL_HAIRPIN_IDXS, N_SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_START_POS
from common.utils import matching_to_db
from common.utils import MAX_PRECOMPUTE, MAX_LOOP
from common import brute_force
from common import nussinov as nus

from common.utils import get_rand_seq, seq_to_one_hot



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
                    bkm1_sm = jnp.sum(vmap(bkm1_fn)(N4))l

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
                bi_bj_sm = base_en*P[bi, bj, i, j]*em.en_ext_branch(bi, bj)

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
                bip1_bjm1_sm = vmap(vmap(bip1_bjm1_fn, (0, None)), (None, 0))(N4, N4)

                bi_bj_sm += bip1_sm + bjm1_sm + bip1_bjm1_sm
                return bi_bj_sm

            j_sm = jnp.sum(vmap(vmap(bi_bj_fn, (0, None)), (None, 0))(N4, N4))
            return jnp.where(cond, j_sm, 0.0)

        j_sm = jnp.sum(vmap(j_fn)(jnp.arange(seq_len+1)))
        sm = j_sm + E[i+1]

        return E.at[i].set(sm)

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
            # P = fill_paired(i, padded_p_seq, OMM, ML, P)
            ML = fill_multi(i, padded_p_seq, ML, P)
            E = fill_external(i, padded_p_seq, P, E)

            return (OMM, P, ML, E), None

        (OMM, P, ML, E), _ = scan(fill_table,
                                  (OMM, P, ML, E),
                                  jnp.arange(seq_len, 0, -1))

        return E[1]

    return ss_partition
