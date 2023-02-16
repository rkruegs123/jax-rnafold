import numpy as np
from common.rna_utils import RNA_ALPHA, SPECIAL_HAIRPINS, INVALID_BASE, structure_tree
from common.energy_hash import float_hash


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

    def en_internal_asym(self, asym):
        pass

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        en = self.en_internal_init(lup+rup)*self.en_internal_asym(abs(lup - rup))*self.en_il_inner_mismatch(
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

    def en_internal_asym(self, asym):
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

    def en_internal_asym(self, asym):
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

    def en_internal_asym(self, asym):
        return self.hash(asym, 17)

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
    dp = np.zeros((2, 2, n+1))
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
