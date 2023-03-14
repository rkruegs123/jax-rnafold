import numpy as np
import unittest

from jax_rnafold.common.utils import NTS, NBPS, HAIRPIN, INVALID_BASE, get_bp_bases
from jax_rnafold.common.utils import random_pseq, matching_to_db
from jax_rnafold.common import brute_force
from jax_rnafold.d1 import energy
from jax_rnafold.common.partition import psum_hairpin


def ss_partition(p_seq, em: energy.Model, dtype=np.float64):
    n = p_seq.shape[0]

    # Pad the ends with 1s to allow indexing off the end to always be a valid base
    # Because of this, we index p_seq using 1-based indexing
    padded_p_seq = np.zeros((n+2, 4), dtype=dtype)
    for i in range(n):
        padded_p_seq[i+1] = p_seq[i]
    # Base at 0 and n+1 is always the 0-th base
    padded_p_seq[n+1] = np.zeros(4, dtype=dtype)
    padded_p_seq[n+1, 0] = 1
    padded_p_seq[0] = np.zeros(4, dtype=dtype)
    padded_p_seq[0, 0] = 1

    p_seq = padded_p_seq

    # Pad on the right by 1
    E = np.zeros((n+2), dtype=dtype)
    P = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)
    ML = np.zeros((3, n+2, n+2), dtype=dtype)
    OMM = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)
    E[n+1] = 1
    ML[0, :, :] = 1

    def fill_external(i):
        sm = E[i+1]
        for j in range(i+1, n+1):
            for bi in range(NTS):
                for bj in range(NTS):
                    base_en = E[j+1]*p_seq[i, bi]*p_seq[j, bj]
                    sm += base_en*P[bi, bj, i, j]*em.en_ext_branch(bi, bj)
                    for bip1 in range(NTS):
                        sm += base_en*P[bip1, bj, i+1, j]*em.en_ext_branch(
                            bip1, bj)*p_seq[i+1, bip1]*em.en_5dangle(bi, bip1, bj)
                    for bjm1 in range(NTS):
                        sm += base_en*P[bi, bjm1, i, j-1]*em.en_ext_branch(
                            bi, bjm1)*p_seq[j-1, bjm1]*em.en_3dangle(bi, bjm1, bj)
                    for bip1 in range(NTS):
                        for bjm1 in range(NTS):
                            sm += base_en*P[bip1, bjm1, i+1, j-1]*em.en_ext_branch(
                                bip1, bjm1)*p_seq[j-1, bjm1]*p_seq[i+1, bip1]*em.en_term_mismatch(bi, bip1, bjm1, bj)
        E[i] = sm

    def fill_multi(i):
        for nb in range(3):
            for j in range(i, n+1):
                sm = ML[nb, i+1, j]
                for k in range(i, j+1):
                    for bi in range(NTS):
                        for bk in range(NTS):
                            base_en = ML[max(0, nb-1), k+1, j] * \
                                p_seq[i, bi]*p_seq[k, bk]
                            sm += base_en*P[bi, bk, i, k] * \
                                em.en_multi_branch(bi, bk)
                            for bip1 in range(NTS):
                                sm += base_en*P[bip1, bk, i+1, k]*em.en_multi_branch(
                                    bip1, bk)*p_seq[i+1, bip1]*em.en_5dangle(bi, bip1, bk)
                            for bkm1 in range(NTS):
                                sm += base_en*P[bi, bkm1, i, k-1]*em.en_multi_branch(
                                    bi, bkm1)*p_seq[k-1, bkm1]*em.en_3dangle(bi, bkm1, bk)
                            for bip1 in range(NTS):
                                for bkm1 in range(NTS):
                                    sm += base_en*P[bip1, bkm1, i+1, k-1]*em.en_multi_branch(
                                        bip1, bkm1)*p_seq[i+1, bip1]*p_seq[k-1, bkm1]*em.en_term_mismatch(bi, bip1, bkm1, bk)
                ML[nb, i, j] = sm

    def fill_outer_mismatch(k):
        for l in range(k+1, n):
            for bpkl in range(NBPS):
                bk, bl = get_bp_bases(bpkl)
                for bkm1 in range(NTS):
                    for blp1 in range(NTS):
                        OMM[bk, bl, k, l] += em.en_il_outer_mismatch(bk, bl, bkm1,
                                                                     blp1)*p_seq[k-1, bkm1]*p_seq[l+1, blp1]

    def psum_internal_loops(bi, bj, i, j):
        sm = 0
        mmij = 0
        for bip1 in range(NTS):
            for bjm1 in range(NTS):
                mmij += p_seq[i+1, bip1]*p_seq[j-1, bjm1] * \
                    em.en_il_inner_mismatch(bi, bj, bip1, bjm1)
        for bpkl in range(NBPS):
            bk, bl = get_bp_bases(bpkl)
            # Both of these loops can be factored out
            for bip1 in range(NTS):
                for bjm1 in range(NTS):
                    pr_ij_mm = p_seq[i+1, bip1]*p_seq[j-1, bjm1]
                    # 1x1. Don't need safe_P since we pad on both sides.
                    sm += P[bk, bl, i+2, j-2]*p_seq[i+2, bk]*p_seq[j-2, bl]*pr_ij_mm * \
                        em.en_internal(bi, bj, bk, bl, bip1,
                                       bjm1, bip1, bjm1, 1, 1)

                    # 1xn (n>=2)
                    for z in range(i+3, j-2):
                        # This loop could be optimised with the mismatch trick.
                        # Probably not worth it.
                        for b in range(NTS):
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-z-1)
                            sm += P[bk, bl, i+2, z]*p_seq[i+2, bk] * \
                                p_seq[z, bl]*p_seq[z+1, b]*pr_ij_mm*il_en
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, b, bjm1, z-i-1, 1)
                            sm += P[bk, bl, z, j-2]*p_seq[z, bk] * \
                                p_seq[j-2, bl]*p_seq[z-1, b]*pr_ij_mm*il_en
            # other internal loops
            for k in range(i+2, j-2):
                for l in range(k+1, j-1):
                    lup, rup = k-i-1, j-l-1
                    # Special cases. Can be replaced with wheres.
                    if lup <= 1 or rup <= 1:
                        # 1xn already done
                        continue
                    if lup == 2 and rup == 2 or lup == 2 and rup == 3 or lup == 3 and rup == 2:
                        # 2x2, 2x3, 3x2
                        # Could be optimised using the mismatch trick.
                        # Probably not worth it.
                        for bip1 in range(NTS):
                            for bjm1 in range(NTS):
                                for bkm1 in range(NTS):
                                    for blp1 in range(NTS):
                                        sm += P[bk, bl, k, l]*p_seq[k, bk]*p_seq[l, bl]*em.en_internal(
                                            bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*p_seq[k-1, bkm1]*p_seq[l+1, blp1]*p_seq[i+1, bip1]*p_seq[j-1, bjm1]
                    else:
                        # Optimised using the mismatch trick (mmij*OMM)
                        init_and_pair = em.en_internal_init(
                            lup+rup)*em.en_internal_asym(lup, rup)*P[bk, bl, k, l]*p_seq[k, bk]*p_seq[l, bl]
                        sm += OMM[bk, bl, k, l]*mmij*init_and_pair
        return sm

    def psum_bulges(bi, bj, i, j):
        sm = 0
        for bpkl in range(NBPS):
            bk, bl = get_bp_bases(bpkl)
            for kl in range(i+2, j-1):
                sm += P[bk, bl, i+1, kl]*p_seq[i+1, bk] * \
                    p_seq[kl, bl]*em.en_bulge(bi, bj, bk, bl, j-kl-1)
                sm += P[bk, bl, kl, j-1]*p_seq[kl, bk] * \
                    p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, kl-i-1)
        return sm

    def psum_multiloops(bi, bj, i, j):
        closing_en = em.en_multi_closing(bi, bj)
        sm = closing_en*ML[2, i+1, j-1]
        for bip1 in range(NTS):
            sm += closing_en*ML[2, i+2, j-1]*p_seq[i+1,
                                                   bip1]*em.en_3dangle_inner(bi, bip1, bj)
        for bjm1 in range(NTS):
            sm += closing_en*ML[2, i+1, j-2]*p_seq[j-1,
                                                   bjm1]*em.en_5dangle_inner(bi, bjm1, bj)
        for bip1 in range(NTS):
            for bjm1 in range(NTS):
                sm += closing_en*ML[2, i+2, j-2]*p_seq[i+1, bip1] * \
                    p_seq[j-1, bjm1] * \
                    em.en_term_mismatch_inner(bi, bip1, bjm1, bj)
        return sm

    def fill_paired(i):
        for bpij in range(NBPS):
            for j in range(i+HAIRPIN+1, n+1):
                bi, bj = get_bp_bases(bpij)
                sm = psum_hairpin(p_seq, em, bi, bj, i, j)
                sm += psum_bulges(bi, bj, i, j)
                sm += psum_internal_loops(bi, bj, i, j)
                # Stacks
                for bpkl in range(NBPS):
                    bk, bl = get_bp_bases(bpkl)
                    sm += P[bk, bl, i+1, j-1]*p_seq[i+1, bk] * \
                        p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)
                # Multi-loops
                sm += psum_multiloops(bi, bj, i, j)
                P[bi, bj, i, j] = sm

    for i in range(n, 0, -1):
        fill_outer_mismatch(i)
        fill_paired(i)
        fill_multi(i)
        fill_external(i)

    return E[1]


class TestSSPartitionFunction(unittest.TestCase):
    def _random_seq_test(self, n, em):
        p_seq = random_pseq(n)
        q = ss_partition(p_seq, em)
        brute_q = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, matching_to_db(match), em))
        print(n, brute_q, q)
        self.assertAlmostEqual(brute_q, q, places=7)

    def test_all_1_model_to_10(self):
        for n in range(1, 10):
            self._random_seq_test(n, energy.All1Model())

    def test_random_model_to_10(self):
        for n in range(1, 10):
            self._random_seq_test(n, energy.RandomModel())
