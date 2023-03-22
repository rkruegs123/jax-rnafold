import numpy as np
import unittest

from jax_rnafold.common.utils import NTS, NBPS, HAIRPIN, get_bp_bases, structure_tree
from jax_rnafold.common.utils import matching_2_dot_bracket, random_pseq
from jax_rnafold.d1 import energy
from jax_rnafold.common.partition import psum_hairpin
from jax_rnafold.common import brute_force, sampling


def seq_partition(p_seq, db, em: energy.Model, dtype=np.float64):
    n = len(db)
    ch, right = structure_tree(db)

    ch_idx = [-1]*n
    num_c = [0]*n
    children = []
    external_c = []

    # The tree needs to be flattened for Jax JIT to work
    # Also, the bottom up order is required for the DP to work
    order = []

    def rec(atl):
        if atl not in ch:
            order.append(atl)
            return
        ch_idx[atl] = len(children)
        num_c[atl] = len(ch[atl])
        children.extend(ch[atl])
        for cl in ch[atl]:
            rec(cl)
        # atl must come after all of its children
        order.append(atl)

    for k in ch[-1]:
        external_c.append(k)
        rec(k)

    # order, p_seq, right, ch_idx, num_c, and children can all be precomputed and saved at this point

    dp = np.zeros((NBPS, n), dtype=dtype)

    def psum_kloop(st, en, closing_pair):
        left = children if closing_pair[0] != -1 else external_c
        n = en-st

        def branch(idx):
            i = left[st+idx]
            return i, right[i]
        en_branch = em.en_ext_branch if closing_pair[0] == - \
            1 else em.en_multi_branch

        def next_i(idx):
            if idx == n-1:
                return closing_pair[1]
            return branch(idx+1)[0]

        # kdp[last, curr, b] is the pr weighted sum of the subloop from from b to the end
        # Curr is true if the 5' dangle is usable
        # Last is true if the last branch can 3' dangle
        kdp = np.zeros((2, 2, n+1))
        kdp[:, :, n] = 1
        for b in range(n-1, -1, -1):
            for last in range(2):
                for curr in range(2):
                    i, j = branch(b)
                    nexti = next_i(b)
                    for bij in range(NBPS):
                        bi, bj = get_bp_bases(bij)
                        base = en_branch(
                            bi, bj) * p_seq[i, bi] * p_seq[j, bj] * dp[bij, left[st+b]]
                        # Normal branch
                        kdp[last, curr, b] += kdp[last,
                                                  int(nexti > j+1), b+1] * base
                        if curr == 1:  # 5' dangle
                            for bim1 in range(NTS):
                                kdp[last, curr, b] += kdp[last, int(nexti > j+1), b+1] * em.en_5dangle(
                                    bim1, bi, bj) * p_seq[i-1, bim1] * base
                        if b < n-1 or last:
                            # Edge case where last is 1 but there is no last unpaired nt
                            if j+1 >= closing_pair[1]:
                                continue
                            # No room for 3' dangle before next branch
                            if b < n-1 and nexti == j+1:
                                continue
                            # 3' dangle
                            for bjp1 in range(NTS):
                                kdp[last, curr, b] += kdp[last, int(nexti > j+2), b+1]*em.en_3dangle(
                                    bi, bj, bjp1)*p_seq[j+1, bjp1]*base
                                if curr == 1:  # Terminal mismatch
                                    for bim1 in range(NTS):
                                        kdp[last, curr, b] += kdp[last, int(nexti > j+2), b+1]*em.en_term_mismatch(
                                            bim1, bi, bj, bjp1)*p_seq[i-1, bim1]*p_seq[j+1, bjp1]*base
        return kdp
    # Must be a scan because the order is important
    for i in order:
        j = right[i]
        if num_c[i] > 1:  # Only compute kdp for multiloops
            kdp = psum_kloop(ch_idx[i], ch_idx[i]+num_c[i], (i, j))
        for bij in range(NBPS):
            bi, bj = get_bp_bases(bij)
            boltz = 0
            if num_c[i] == 0:  # Hairpin loop
                boltz += psum_hairpin(p_seq, em, bi, bj, i, j)
            elif num_c[i] == 1:  # Two loop
                k = children[ch_idx[i]]
                l = right[k]
                for bkl in range(NBPS):
                    bk, bl = get_bp_bases(bkl)
                    if k == i+1 and l == j-1:  # Stack
                        boltz += em.en_stack(bi, bj, bk, bl) * \
                            p_seq[k, bk]*p_seq[l, bl]*dp[bkl, k]
                    elif k == i+1 or l == j-1:  # Bulge loop
                        nunpaired = max(k-i-1, j-l-1)
                        boltz += em.en_bulge(bi, bj, bk, bl, nunpaired) * \
                            p_seq[k, bk]*p_seq[l, bl]*dp[bkl, k]
                    else:  # Internal loop
                        lup = k-i-1
                        rup = j-l-1
                        for bip1 in range(NTS):
                            for bjm1 in range(NTS):
                                for bkm1 in range(NTS):
                                    for blp1 in range(NTS):
                                        if lup == 1 and bip1 != bkm1:
                                            continue
                                        if rup == 1 and bjm1 != blp1:
                                            continue
                                        pr = p_seq[i+1, bip1] * p_seq[j-1,
                                                                      bjm1] * p_seq[k, bk] * p_seq[l, bl]
                                        if lup > 1:
                                            pr *= p_seq[k-1, bkm1]
                                        if rup > 1:
                                            pr *= p_seq[l+1, blp1]
                                        boltz += em.en_internal(
                                            bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) * pr * dp[bkl, k]
            else:  # Multiloop
                # No CTD case
                lj = right[children[ch_idx[i]+num_c[i]-1]]
                fi = children[ch_idx[i]]
                sm = kdp[int(lj < j-1), int(i+1 < fi), 0]
                if i+1 < fi:
                    for bip1 in range(NTS):
                        sm += kdp[int(lj < j-1), int(i+2 < fi), 0] * \
                            em.en_3dangle_inner(
                                bi, bip1, bj) * p_seq[i+1, bip1]
                if lj < j-1:
                    for bjm1 in range(NTS):
                        sm += kdp[int(lj < j-2), int(i+1 < fi), 0] * \
                            em.en_5dangle_inner(
                                bi, bjm1, bj) * p_seq[j-1, bjm1]
                        if i+1 < fi:
                            for bip1 in range(NTS):
                                sm += kdp[int(lj < j-2), int(i+2 < fi), 0] * em.en_term_mismatch_inner(
                                    bi, bip1, bjm1, bj) * p_seq[i+1, bip1] * p_seq[j-1, bjm1]
                sm *= em.en_multi_closing(bi, bj)
                boltz += sm
            dp[bij, i] = boltz
    # External loop
    kdp = psum_kloop(0, len(external_c), (-1, n))
    boltz = 1
    if len(external_c) > 0:
        boltz = kdp[int(right[external_c[-1]]+1 < n),
                    int(external_c[0] > 0), 0]
    return boltz


class TestSeqPartitionFunction(unittest.TestCase):
    def _run_test(self, p_seq, db, em):
        ans = seq_partition(p_seq, db, em)
        ref_ans = brute_force.seq_partition(
            p_seq, db, energy_fn=lambda seq, db: energy.calculate(seq, db, em))
        print(db, ans, ref_ans)
        self.assertAlmostEqual(ans, ref_ans, places=7)

    def _equiprobable_p_seq(self, n):
        return np.full((n, 4), 0.25)

    def test_handmade(self):
        dbs = ["(..)", "(.).", ".(.)", ".().", "()()", ".().().", "((..))", "((()))", "(..())", "(..().)",
               "(.().)", "(..()..)", "(()())", "(.()())", "(()().)", "(.().().)", "((()()))", "((().()..))"]
        all1_em = energy.All1Model()
        random_em = energy.RandomModel()
        for db in dbs:
            self._run_test(self._equiprobable_p_seq(len(db)), db, all1_em)
            self._run_test(self._equiprobable_p_seq(len(db)), db, random_em)

    def test_random(self):
        for _ in range(5):
            for n in range(1, 12):
                p_seq = random_pseq(n)
                uss = sampling.UniformStructureSampler()
                uss.precomp([None]*n)
                db = matching_2_dot_bracket(uss.get_nth(
                    np.random.randint(uss.count_structures())))
                self._run_test(p_seq, db, energy.RandomModel())
