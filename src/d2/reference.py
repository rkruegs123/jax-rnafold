import unittest
from common.utils import ALL_PAIRS, RNA_ALPHA, NTS, NBPS, HAIRPIN, SPECIAL_HAIRPINS, INVALID_BASE
from d2 import energy
import numpy as np
from common import utils
from common import brute_force


def get_bp_bases(bp):
    return (RNA_ALPHA.index(ALL_PAIRS[bp][0]), RNA_ALPHA.index(ALL_PAIRS[bp][1]))


def psum_hairpin_not_special(p_seq, em, bi, bj, i, j):
    # Special case for HAIRPIN<=1
    # Necessary to respect conditional probability the mismatch
    # Can be removed or made branchless/jax.where
    if i+1 == j:
        return em.en_hairpin_not_special(bi, bj, bj, bi, 0)
    sm = 0
    if i+1 == j-1:
        for bip1 in range(NTS):
            sm += p_seq[i+1, bip1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bip1, 1)
        return sm
    for bip1 in range(NTS):
        for bjm1 in range(NTS):
            sm += p_seq[i+1, bip1]*p_seq[j-1, bjm1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bjm1, j-i-1)
    return sm


def pr_special_hairpin(p_seq, id, i, j):
    pr = 1
    for k in range(i+1, j):
        # .index can be replace with a table
        pr *= p_seq[k, RNA_ALPHA.index(SPECIAL_HAIRPINS[id][k-i])]
    return pr


def psum_hairpin_special(p_seq, em, bi, bj, i, j):
    sm = 0
    for id in range(len(SPECIAL_HAIRPINS)):
        # Can be made branchless
        if SPECIAL_HAIRPINS[id][0] != RNA_ALPHA[bi]:
            continue
        if SPECIAL_HAIRPINS[id][-1] != RNA_ALPHA[bj]:
            continue
        if len(SPECIAL_HAIRPINS[id]) != j-i+1:
            continue
        sm += pr_special_hairpin(p_seq, id, i, j) * em.en_hairpin_special(id)
    return sm


def psum_hairpin_special_correction(p_seq, em, bi, bj, i, j):
    sm = 0
    for id in range(len(SPECIAL_HAIRPINS)):
        # Can be made branchless
        if SPECIAL_HAIRPINS[id][0] != RNA_ALPHA[bi]:
            continue
        if SPECIAL_HAIRPINS[id][-1] != RNA_ALPHA[bj]:
            continue
        if len(SPECIAL_HAIRPINS[id]) != j-i+1:
            continue
        # .index can be replace with a table
        bip1 = RNA_ALPHA.index(SPECIAL_HAIRPINS[id][1])
        bjm1 = RNA_ALPHA.index(SPECIAL_HAIRPINS[id][-2])
        sm += pr_special_hairpin(p_seq, id, i, j) * em.en_hairpin_not_special(
            bi, bj, bip1, bjm1, len(SPECIAL_HAIRPINS[id])-2)
    return sm


def psum_hairpin(p_seq, em, bi, bj, i, j):
    return psum_hairpin_not_special(p_seq, em, bi, bj, i, j) + psum_hairpin_special(p_seq, em, bi, bj, i, j) - psum_hairpin_special_correction(p_seq, em, bi, bj, i, j)


def seq_partition(p_seq, db, em: energy.Model, dtype=np.float64):
    # Very cheesy hack to get a fast(er) O(N^2) sequence partition function
    # Linear time is possible but who cares
    n = len(db)

    # This is all precomp that happens before compilation
    match = [i for i in range(n+1)]
    stk = []
    for i in range(n):
        if db[i] == '(':
            stk.append(i)
        elif db[i] == ')':
            j = stk.pop()
            match[j+1] = i+1
    # Precomp unpaired ranges
    up = np.ones((n+2, n+2), dtype=bool)
    for i in range(n+1):
        for j in range(i+2, n+1):
            up[i, j] = (db[j-2]=='.')*up[i, j-1]

    padded_p_seq = np.zeros((n+2, 4), dtype=dtype)
    for i in range(n):
        padded_p_seq[i+1] = p_seq[i]
    # Base at 0 and n+1 is always the 0-th base
    padded_p_seq[n+1] = np.zeros(4, dtype=dtype)
    padded_p_seq[n+1, 0] = 1
    padded_p_seq[0] = np.zeros(4, dtype=dtype)
    padded_p_seq[0, 0] = 1

    p_seq = padded_p_seq

    E = np.zeros((NTS, NTS, n+2), dtype=dtype)
    P = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)
    ML = np.zeros((NTS, NTS, NTS, NTS, 3, n+2, n+2), dtype=dtype)
    OMM = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)

    def fill_external(i):
        for bim1 in range(NTS):
            for bi in range(NTS):
                sm = 0
                j = match[i]
                # print(i, j)
                if j == i:
                    for bip1 in range(NTS):
                        sm += (E[bi, bip1, i+1] + int(i == n))*p_seq[i+1, bip1]
                else:
                    for bj in range(NTS):
                        for bjp1 in range(NTS):
                            # These can be where instead. Currently branchless arithmetic.
                            dangle5 = (i == 1)*INVALID_BASE + (i != 1)*bim1
                            dangle3 = (j == n)*INVALID_BASE + (j != n)*bjp1
                            # print(bi, bj, i,j,P[bi, bj, i, j])
                            sm += P[bi, bj, i, j]*(E[bj, bjp1, j+1] + int(j == n))*em.en_ext_branch(
                                dangle5, bi, bj, dangle3)*p_seq[j, bj]*p_seq[j+1, bjp1]
                E[bim1, bi, i] = sm


    def fill_multi(i):
        for bim1 in range(NTS):
            for bi in range(NTS):
                for bj in range(NTS):
                    for bjp1 in range(NTS):
                        for nb in range(3):
                            for j in range(i, n+1):
                                sm = 0
                                k = match[i]
                                # Replace if/elif/else with where
                                if k == i:
                                    for bip1 in range(NTS):
                                        sm += (ML[bi, bip1, bj, bjp1, nb, i+1,
                                                  j]+int(i+1 > j and nb == 0)) * p_seq[i+1, bip1]
                                elif k == j:
                                    # Special case for k==j
                                    sm += P[bi, bj, i, j] * \
                                        int(nb <= 1) * \
                                        em.en_multi_branch(bim1, bi, bj, bjp1)
                                elif k == j-1:
                                    for bk in range(NTS):
                                        # Special case for k==j-1
                                        sm += P[bi, bk, i, j-1]*int(nb <= 1) * em.en_multi_branch(
                                            bim1, bi, bk, bj)*p_seq[j-1, bk]
                                else:
                                    for bk in range(NTS):
                                        for bkp1 in range(NTS):
                                            # The max function can be made branchless
                                            sm += P[bi, bk, i, k]*ML[bk, bkp1, bj, bjp1, max(0, nb-1), k+1, j]*em.en_multi_branch(
                                                bim1, bi, bk, bkp1)*p_seq[k, bk]*p_seq[k+1, bkp1]
                                ML[bim1, bi, bj, bjp1, nb, i, j] = sm

    def fill_outer_mismatch(k):
        for l in range(k+1, n+1):
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
                                       bjm1, bip1, bjm1, 1, 1)*up[i, i+2]*up[j-2, j]

                    # 1xn (n>=2)
                    for z in range(i+3, j-2):
                        # This loop could be optimised with the mismatch trick.
                        # Probably not worth it.
                        for b in range(NTS):
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-z-1)
                            sm += P[bk, bl, i+2, z]*p_seq[i+2, bk] * \
                                p_seq[z, bl]*p_seq[z+1, b] * \
                                pr_ij_mm*il_en*up[i, i+2]*up[z, j]
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, b, bjm1, z-i-1, 1)
                            sm += P[bk, bl, z, j-2]*p_seq[z, bk] * \
                                p_seq[j-2, bl]*p_seq[z-1, b] * \
                                pr_ij_mm*il_en*up[i, z]*up[j-2, j]
            # other internal loops
            for k in range(i+2, j-2):
                l = match[k]
                actually_paired = l > k
                # Replace with where
                # Avoids out of bounds indexing
                if not actually_paired:
                    l = j-1
                res = 0
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
                                    res += P[bk, bl, k, l]*p_seq[k, bk]*p_seq[l, bl]*em.en_internal(
                                        bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*p_seq[k-1, bkm1]*p_seq[l+1, blp1]*p_seq[i+1, bip1]*p_seq[j-1, bjm1]*up[i, k]*up[l, j]
                else:
                    # Optimised using the mismatch trick (mmij*OMM)
                    init_and_pair = em.en_internal_init(
                        lup+rup)*em.en_internal_asym(lup, rup)*P[bk, bl, k, l]*p_seq[k, bk]*p_seq[l, bl]
                    res += OMM[bk, bl, k, l]*mmij * \
                        init_and_pair*up[i, k]*up[l, j]
                # Replace with where
                if actually_paired:
                    sm += res
        return sm

    def psum_bulges(bi, bj, i, j):
        sm = 0
        for bpkl in range(NBPS):
            bk, bl = get_bp_bases(bpkl)
            for kl in range(i+2, j-1):
                sm += P[bk, bl, i+1, kl]*p_seq[i+1, bk] * \
                    p_seq[kl, bl]*em.en_bulge(bi, bj, bk, bl, j-kl-1)*up[kl, j]
                sm += P[bk, bl, kl, j-1]*p_seq[kl, bk] * \
                    p_seq[j-1, bl] * \
                    em.en_bulge(bi, bj, bk, bl, kl-i-1)*up[i, kl]
        return sm

    def fill_paired(i):
        j = match[i]
        # Replace with cond (not where)
        # I think you can actually do this with a cond because it's in the outer scan
        if j == i:
            return
        for bpij in range(NBPS):
            bi, bj = get_bp_bases(bpij)
            sm = psum_hairpin(p_seq, em, bi, bj, i, j)*up[i, j]
            sm += psum_bulges(bi, bj, i, j)
            sm += psum_internal_loops(bi, bj, i, j)
            # Stacks
            for bpkl in range(NBPS):
                bk, bl = get_bp_bases(bpkl)
                sm += P[bk, bl, i+1, j-1]*p_seq[i+1, bk] * \
                    p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)
            # Multi-loops
            for bip1 in range(NTS):
                for bjm1 in range(NTS):
                    sm += ML[bi, bip1, bjm1, bj, 2, i+1, j-1]*p_seq[i+1, bip1] * \
                        p_seq[j-1, bjm1] * \
                        em.en_multi_closing(bi, bip1, bjm1, bj)
            P[bi, bj, i, j] = sm

    for i in range(n, 0, -1):
        fill_outer_mismatch(i)
        fill_paired(i)
        fill_multi(i)
        fill_external(i)

    sm = 0
    for bim1 in range(NTS):
        for bi in range(NTS):
            sm += E[bim1, bi, 1]*p_seq[1, bi]*p_seq[0, bim1]
    return sm


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

    E = np.zeros((NTS, NTS, n+2), dtype=dtype)
    P = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)
    ML = np.zeros((NTS, NTS, NTS, NTS, 3, n+2, n+2), dtype=dtype)
    OMM = np.zeros((NTS, NTS, n+2, n+2), dtype=dtype)

    def fill_external(i):
        for bim1 in range(NTS):
            for bi in range(NTS):
                sm = 0
                for bip1 in range(NTS):
                    sm += (E[bi, bip1, i+1] + int(i == n))*p_seq[i+1, bip1]
                for j in range(i+1, n+1):
                    for bj in range(NTS):
                        for bjp1 in range(NTS):
                            # These can be where instead. Currently branchless arithmetic.
                            dangle5 = (i == 1)*INVALID_BASE + (i != 1)*bim1
                            dangle3 = (j == n)*INVALID_BASE + (j != n)*bjp1
                            sm += P[bi, bj, i, j]*(E[bj, bjp1, j+1] + int(j == n))*em.en_ext_branch(
                                dangle5, bi, bj, dangle3)*p_seq[j, bj]*p_seq[j+1, bjp1]
                E[bim1, bi, i] = sm

    def fill_multi(i):
        for bim1 in range(NTS):
            for bi in range(NTS):
                for bj in range(NTS):
                    for bjp1 in range(NTS):
                        for nb in range(3):
                            for j in range(i, n+1):
                                sm = 0
                                for bip1 in range(NTS):
                                    sm += (ML[bi, bip1, bj, bjp1, nb, i+1,
                                           j]+int(i+1 > j and nb == 0)) * p_seq[i+1, bip1]
                                # Special case for k==j
                                sm += P[bi, bj, i, j]*int(nb <= 1) * \
                                    em.en_multi_branch(bim1, bi, bj, bjp1)
                                for bk in range(NTS):
                                    # Special case for k==j-1
                                    sm += P[bi, bk, i, j-1]*int(nb <= 1) * em.en_multi_branch(
                                        bim1, bi, bk, bj)*p_seq[j-1, bk]
                                for k in range(i, j-1):
                                    # A minor optimisation is to only use valid bk pairs for bi
                                    for bk in range(NTS):
                                        for bkp1 in range(NTS):
                                            # The max function can be made branchless
                                            sm += P[bi, bk, i, k]*ML[bk, bkp1, bj, bjp1, max(0, nb-1), k+1, j]*em.en_multi_branch(
                                                bim1, bi, bk, bkp1)*p_seq[k, bk]*p_seq[k+1, bkp1]
                                ML[bim1, bi, bj, bjp1, nb, i, j] = sm

    def fill_outer_mismatch(k):
        for l in range(k+1, n+1):
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
                for bip1 in range(NTS):
                    for bjm1 in range(NTS):
                        sm += ML[bi, bip1, bjm1, bj, 2, i+1, j-1]*p_seq[i+1, bip1] * \
                            p_seq[j-1, bjm1] * \
                            em.en_multi_closing(bi, bip1, bjm1, bj)
                P[bi, bj, i, j] = sm

    for i in range(n, 0, -1):
        fill_outer_mismatch(i)
        fill_paired(i)
        fill_multi(i)
        fill_external(i)

    sm = 0
    for bim1 in range(NTS):
        for bi in range(NTS):
            sm += E[bim1, bi, 1]*p_seq[1, bi]*p_seq[0, bim1]
    return sm


class TestPartitionFunction(unittest.TestCase):
    def _random_p_seq(self, n):
        p_seq = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            p_seq[i] = np.random.random_sample(4)
            p_seq[i] /= np.sum(p_seq[i])
        return p_seq

    def _all_1_test(self, n):
        import common.nussinov as nus
        em = energy.All1Model()
        p_seq = self._random_p_seq(n)
        nuss = nus.ss_partition(p_seq, en_pair=nus.en_pair_1)
        vien = ss_partition(p_seq, em)
        print(n, nuss, vien)
        self.assertAlmostEqual(nuss, vien, places=7)

    def _brute_model_test(self, em, n):
        p_seq = self._random_p_seq(n)
        vien = ss_partition(p_seq, em)
        brute = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, utils.matching_to_db(match), em))
        print(n, brute, vien)
        self.assertAlmostEqual(brute, vien, places=7)

    def _random_test(self, n):
        em = energy.RandomModel()
        self._brute_model_test(em, n)

    def test_all_1_model_to_10(self):
        for n in range(1, 10):
            self._all_1_test(n)

    def test_all_1_model_12(self):
        # First multiloop for HAIRPIN=3
        self._all_1_test(12)

    def test_all_1_model_20(self):
        self._all_1_test(20)

    def test_random_model_to_10(self):
        for n in range(1, 10):
            self._random_test(n)

    def test_seq_one_hot(self):
        from common import sampling
        import random
        for n in range(1, 20):
            seq = utils.random_primary(n)
            p_seq = utils.one_hot_seq(seq)
            uss = sampling.UniformStructureSampler()
            uss.precomp(seq)
            db = utils.matching_to_db(uss.get_nth(
                random.randrange(0, uss.count_structures())))
            em = energy.RandomModel()
            e_calc = energy.calculate(seq, db, em)
            e_spart = seq_partition(p_seq, db, em)
            print(n, seq, db, e_calc, e_spart)
            self.assertAlmostEqual(e_calc, e_spart, places=7)

    def test_seq_brute(self):
        from common import sampling
        import random
        for it in range(5):
            for n in range(1, 11):
                p_seq = self._random_p_seq(n)
                uss = sampling.UniformStructureSampler()
                uss.precomp([None]*n)
                db = utils.matching_to_db(uss.get_nth(random.randrange(0, uss.count_structures())))
                em = energy.RandomModel()
                e_brute = brute_force.seq_partition(p_seq, db, lambda seq, db: energy.calculate(seq, db, em))
                e_spart = seq_partition(p_seq, db, em)
                print(n, db, e_brute, e_spart)
                self.assertAlmostEqual(e_brute, e_spart, places=7)

    # def test(self):
    #     for _ in range(50):
    #         self._brute_model_test(energy.RandomModel(), 10)


if __name__ == '__main__':
    unittest.main()
