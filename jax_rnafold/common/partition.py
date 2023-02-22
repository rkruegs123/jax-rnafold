from jax_rnafold.common.utils import RNA_ALPHA, ALL_PAIRS, SPECIAL_HAIRPINS, NTS


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
