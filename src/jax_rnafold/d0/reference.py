import numpy as np
import pdb

from jax_rnafold.common.utils import NTS, NBPS, get_bp_bases, structure_tree, TURNER_2004
from jax_rnafold.d0 import energy
from jax_rnafold.common.partition import psum_hairpin



def ss_partition(p_seq: np.ndarray, em: energy.Model, dtype=np.float64) -> float:
    """
    A reference implementation of the generalized McCaskill's
    algorithm for computing the structure-sequence partition
    function.

    Args:
      p_seq: A probabilistic sequence of shape `[n, 4]`.
      em: An energy model.
      dtype: The data type of the DP tables (to set the floating point precision).

    Returns:
      The structure-sequence partition function.
    """
    n = p_seq.shape[0]

    # Pad so out of bounds indexing works on the right
    padded_p_seq = np.zeros((n+1, 4), dtype=dtype)
    for i in range(n):
        padded_p_seq[i] = p_seq[i]
    p_seq = padded_p_seq

    E = np.zeros((n+1), dtype=dtype)
    P = np.zeros((NBPS, n+1, n+1), dtype=dtype)
    ML = np.zeros((3, n+1, n+1), dtype=dtype)
    MB = np.zeros((n+1, n+1), dtype=dtype)
    OMM = np.zeros((n+1, n+1), dtype=dtype)
    E[n] = 1
    ML[0, :, :] = 1

    def fill_external(i):
        sm = E[i+1]
        for j in range(i+1, n):
            for bp in range(NBPS):
                bi, bj = get_bp_bases(bp)
                base_en = E[j+1]*p_seq[i, bi]*p_seq[j, bj]
                sm += base_en*P[bp, i, j]*em.en_ext_branch(bi, bj)
        E[i] = sm

    def fill_multi(i):
        for nb in range(3):
            for j in range(i, n):
                sm = ML[nb, i+1, j]
                for k in range(i, j+1):
                    sm += ML[max(0, nb-1), k+1, j]*MB[i, k]
                ML[nb, i, j] = sm

    def fill_multibranch(i):
        for j in range(i, n+1):
            for bp in range(NBPS):
                bi, bj = get_bp_bases(bp)
                MB[i][j] += P[bp, i, j] * \
                    em.en_multi_branch(bi, bj)*p_seq[i, bi]*p_seq[j, bj]

    def fill_outer_mismatch(i):
        for j in range(i, n):
            for bpij in range(NBPS):
                bi, bj = get_bp_bases(bpij)
                for bim1 in range(NTS):
                    for bjp1 in range(NTS):
                        OMM[i, j] += em.en_il_outer_mismatch(
                            bi, bj, bim1, bjp1)*P[bpij, i, j]*p_seq[i-1, bim1]*p_seq[j+1, bjp1]*p_seq[i, bi]*p_seq[j, bj]

    def psum_internal_loops(bp, i, j):
        bi, bj = get_bp_bases(bp)
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
                    sm += P[bpkl, i+2, j-2]*p_seq[i+2, bk]*p_seq[j-2, bl]*pr_ij_mm * \
                        em.en_internal(bi, bj, bk, bl, bip1,
                                       bjm1, bip1, bjm1, 1, 1)

                    # 1xn (n>=2)
                    # 1x2 and 2x1 can be extracted out as a special case
                    for z in range(i+3, j-2):
                        # This loop could be optimised with the mismatch trick.
                        # Will need a custom OMM table for this.
                        for b in range(NTS):
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, bip1, b, 1, j-z-1)
                            sm += P[bpkl, i+2, z]*p_seq[i+2, bk] * \
                                p_seq[z, bl]*p_seq[z+1, b]*pr_ij_mm*il_en
                            il_en = em.en_internal(
                                bi, bj, bk, bl, bip1, bjm1, b, bjm1, z-i-1, 1)
                            sm += P[bpkl, z, j-2]*p_seq[z, bk] * \
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
                    for bpkl in range(NBPS):
                        bk, bl = get_bp_bases(bpkl)
                        for bip1 in range(NTS):
                            for bjm1 in range(NTS):
                                for bkm1 in range(NTS):
                                    for blp1 in range(NTS):
                                        sm += P[bpkl, k, l]*p_seq[k, bk]*p_seq[l, bl]*em.en_internal(
                                            bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*p_seq[k-1, bkm1]*p_seq[l+1, blp1]*p_seq[i+1, bip1]*p_seq[j-1, bjm1]
                else:
                    # Optimised using the mismatch trick (mmij*OMM)
                    init = em.en_internal_init(
                        lup+rup)*em.en_internal_asym(lup, rup)
                    sm += OMM[k, l]*mmij*init
        return sm

    def psum_bulges(bp, i, j):
        bi, bj = get_bp_bases(bp)
        sm = 0
        for bpkl in range(NBPS):
            bk, bl = get_bp_bases(bpkl)
            for kl in range(i+2, j-1):
                sm += P[bpkl, i+1, kl]*p_seq[i+1, bk] * \
                    p_seq[kl, bl]*em.en_bulge(bi, bj, bk, bl, j-kl-1)
                sm += P[bpkl, kl, j-1]*p_seq[kl, bk] * \
                    p_seq[j-1, bl]*em.en_bulge(bi, bj, bk, bl, kl-i-1)
        return sm

    def fill_paired(i):
        for bpij in range(NBPS):
            for j in range(i+em.hairpin+1, n):
                bi, bj = get_bp_bases(bpij)
                sm = psum_hairpin(p_seq, em, bi, bj, i, j)
                sm += psum_bulges(bpij, i, j)
                sm += psum_internal_loops(bpij, i, j)
                # Stacks
                for bpkl in range(NBPS):
                    bk, bl = get_bp_bases(bpkl)
                    sm += P[bpkl, i+1, j-1]*p_seq[i+1, bk] * \
                        p_seq[j-1, bl]*em.en_stack(bi, bj, bk, bl)
                # Multi-loops
                sm += em.en_multi_closing(bi, bj)*ML[2, i+1, j-1]
                P[bpij, i, j] = sm

    for i in range(n-1, -1, -1):
        fill_paired(i)
        fill_outer_mismatch(i)
        fill_multibranch(i)
        fill_multi(i)
        fill_external(i)

    # print("P: ", P)
    # print("ML: ", ML)

    return E[0]
