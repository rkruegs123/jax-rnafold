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
import jax.debug

from jax_rnafold.d1 import energy
from jax_rnafold.common.checkpoint import checkpoint_scan
from jax_rnafold.common.utils import bp_bases, HAIRPIN, N4, INVALID_BASE, structure_tree, NBPS, N6
from jax_rnafold.common.utils import SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_LENS, \
    SPECIAL_HAIRPIN_IDXS, N_SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_START_POS
from jax_rnafold.common.utils import matching_to_db
from jax_rnafold.common.utils import MAX_PRECOMPUTE, MAX_LOOP
from jax_rnafold.common import brute_force
from jax_rnafold.common import nussinov as nus
from jax_rnafold.common.utils import get_rand_seq, seq_to_one_hot, random_pseq, matching_2_dot_bracket, bcolors, bp_bases

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)


f64 = jnp.float64


checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)

def get_seq_partition_fn(em, db):

    n = len(db)

    if db == len(db) * '.':
        print(f"Warning: structure contains no base pairs, returning a function that always returns 1.0")
        return lambda p_seq: 1.0

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


    # Update to handle external loop nicely
    num_c.append(len(external_c))
    ch_idx.append(len(children))
    children += external_c


    n_external_c = len(external_c)
    external_c = jnp.array(external_c)
    # num_c_max = int(jnp.max(jnp.array(num_c + [n_external_c])))
    num_c_max = int(jnp.max(jnp.array(num_c)))

    right = jnp.array(right)
    order = jnp.array(order)
    num_c = jnp.array(num_c)
    children = jnp.array(children)
    ch_idx = jnp.array(ch_idx)



    # To avoid OOB errors for external loop
    """
    if n_external_c == 0:
        last_external_c = 0.0
        first_external_c = 0.0
    else:
        last_external_c = external_c[-1]
        first_external_c = external_c[0]
    """

    ## End precompute


    @jit
    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1
        u = j - i - 1

        # FIXME: repeated computation should combine with `psum_hairpin_special()`
        @jit
        def pr_special_hairpin(id, i, j):
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            id_len = SPECIAL_HAIRPIN_LENS[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, SPECIAL_HAIRPIN_IDXS[idx_pos]], 1.0)
            ks = jnp.arange(n+1) # FIXME: is this correct?
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        @jit
        def special_hairpin_correction(id):
            sp_hairpin_len = SPECIAL_HAIRPIN_LENS[id]
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (SPECIAL_HAIRPIN_LENS[id] == up2) \
                       & (SPECIAL_HAIRPIN_IDXS[start_pos] == bi) \
                       & (SPECIAL_HAIRPIN_IDXS[end_pos] == bj)

            bjm1 = SPECIAL_HAIRPIN_IDXS[end_pos - 1]
            bip1 = SPECIAL_HAIRPIN_IDXS[start_pos + 1]
            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_not_special(
                                 bi, bj, bip1, bjm1, sp_hairpin_len - 2),
                             0.0)

        summands = vmap(special_hairpin_correction)(jnp.arange(N_SPECIAL_HAIRPINS))
        sm = jnp.sum(summands)
        return sm

    @jit
    def psum_hairpin_special(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1

        @jit
        def pr_special_hairpin(id, i, j):
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            id_len = SPECIAL_HAIRPIN_LENS[id]
            def get_sp_hairpin_nuc_prob(k):
                cond = (k >= i+1) & (k < j)
                idx_pos = start_pos + (k-i)
                return jnp.where(cond, padded_p_seq[k, SPECIAL_HAIRPIN_IDXS[idx_pos]], 1.0)
            ks = jnp.arange(n+1) # FIXME: is this correct?
            prs = vmap(get_sp_hairpin_nuc_prob)(ks)
            pr = 1 # we know i and j match
            pr *= jnp.prod(prs)
            return pr

        @jit
        def special_hairpin(id):
            sp_hairpin_len = SPECIAL_HAIRPIN_LENS[id]
            start_pos = SPECIAL_HAIRPIN_START_POS[id]
            end_pos = start_pos + sp_hairpin_len - 1

            id_valid = (SPECIAL_HAIRPIN_LENS[id] == up2) \
                       & (SPECIAL_HAIRPIN_IDXS[start_pos] == bi) \
                       & (SPECIAL_HAIRPIN_IDXS[end_pos] == bj)

            return jnp.where(id_valid,
                             pr_special_hairpin(id, i, j) * em.en_hairpin_special(id),
                             0.0)

        summands = vmap(special_hairpin)(jnp.arange(N_SPECIAL_HAIRPINS))
        sm = jnp.sum(summands)
        return sm


    @jit
    def psum_hairpin_not_special(bi, bj, i, j, padded_p_seq):
        # Special case for HAIRPIN<=1
        # Necessary to respect conditional probability the mismatch

        u = j-i-1

        @jit
        def u1_fn(bip1):
            return padded_p_seq[i+1, bip1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bip1, 1)
        u1_fn = vmap(u1_fn)

        @jit
        def u_general_fn(bip1, bjm1):
            return padded_p_seq[i+1, bip1]*padded_p_seq[j-1, bjm1] * \
                em.en_hairpin_not_special(bi, bj, bip1, bjm1, j-i-1)
        u_general_fn = vmap(vmap(u_general_fn, (0, None)), (None, 0))


        return jnp.where(u == 0, em.en_hairpin_not_special(bi, bj, bj, bi, 0),
                         jnp.where(u == 1,
                                   jnp.sum(u1_fn(N4)),
                                   jnp.sum(u_general_fn(N4, N4))))

    # FIXME: copied from `ss.py`. Should make the jittable version a global function
    @jit
    def psum_hairpin(bi, bj, i, j, padded_p_seq):
        return psum_hairpin_not_special(bi, bj, i, j, padded_p_seq) \
            + psum_hairpin_special(bi, bj, i, j, padded_p_seq) \
            - psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq)


    @jit
    def psum_twoloop(i, j, bi, bj, p_seq, dp):
        k = children[ch_idx[i]]
        l = right[k]

        lup = k-i-1
        rup = j-l-1

        @jit
        def psum_twoloop_bp(bp_idx):
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]


            @jit
            def internal_case_fn(bip1, bjm1, bkm1, blp1):
                invalid_cond1 = (lup == 1) & (bip1 != bkm1)
                invalid_cond2 = (rup == 1) & (bjm1 != blp1)
                invalid_cond = invalid_cond1 | invalid_cond2 # FIXME: meaning?

                pr = p_seq[i+1, bip1] * p_seq[j-1, bjm1] * p_seq[k, bk] * p_seq[l, bl]
                pr = jnp.where(lup > 1, pr * p_seq[k-1, bkm1], pr)
                pr = jnp.where(rup > 1, pr * p_seq[l+1, blp1], pr)

                return jnp.where(invalid_cond, 0.0,
                                 em.en_internal(bi, bj, bk, bl, bip1, bjm1,
                                                bkm1, blp1, lup, rup) * pr * dp[bp_idx, k])
            internal_fn_mapped = vmap(internal_case_fn, (None, None, None, 0))
            internal_fn_mapped = vmap(internal_fn_mapped, (None, None, 0, None))
            internal_fn_mapped = vmap(internal_fn_mapped, (None, 0, None, None))
            internal_fn_mapped = vmap(internal_fn_mapped, (0, None, None, None))
            internal_sm = jnp.sum(internal_fn_mapped(N4, N4, N4, N4))

            stack_cond = (k == i+1) & (l == j-1)
            bulge_cond = (k == i+1) | (l == j-1)
            bulge_nunpaired = jnp.where(lup > rup, lup, rup) # FIXME: how we want to do branchless max?
            return jnp.where(stack_cond,
                             em.en_stack(bi, bj, bk, bl)*p_seq[k, bk]*p_seq[l, bl]*dp[bp_idx, k],
                             jnp.where(bulge_cond,
                                       em.en_bulge(bi, bj, bk, bl, bulge_nunpaired) * \
                                       p_seq[k, bk]*p_seq[l, bl]*dp[bp_idx, k],
                                       internal_sm))


        bp_vals = vmap(psum_twoloop_bp)(N6)
        return jnp.sum(bp_vals)

    @jit
    def _psum_kloop(i, j, st, en, p_seq, dp):
        # st = ch_idx[i]
        # en = st + num_c[i]
        # j = right[i]

        # left = jnp.where(i != -1, children, external_c) # FIXME: will these always be the same size?
        left = children

        n_c = num_c[i]
        # n = en-st

        # kdp = jnp.zeros((2, 2, n+1))
        kdp = jnp.zeros((2, 2, num_c_max+1))
        kdp = kdp.at[:, :, n_c].set(1)
        # kdp = kdp.at[:, :, n_c].set(1) # Note: I think this would work too

        @jit
        def branch(idx):
            left_i = left[st+idx]
            return left_i, right[left_i]

        @jit
        def next_i(idx):
            return jnp.where(idx == n_c-1, j, branch(idx+1)[0])


        # Note: order of b matters, so we scan
        @jit
        def b_fn(curr_kdp, b):
            k, l = branch(b)
            nexti = next_i(b)

            # Return the value to be *added* to the current entry at [last, curr, b]
            @jit
            def b_last_curr_bp_fn(last, curr, bp_idx):
                bp = bp_bases[bp_idx]
                bk = bp[0]
                bl = bp[1]

                base = jnp.where(i == -1,
                                 em.en_ext_branch(bk, bl),
                                 em.en_multi_branch(bk, bl))
                base = base * p_seq[k, bk] * p_seq[l, bl] * dp[bp_idx, left[st+b]]

                bit = jnp.where(nexti > l+1, 1, 0)
                sm = curr_kdp[last, bit, b+1] * base

                def dangle5_fn(bkm1):
                    return curr_kdp[last, bit, b+1] * em.en_5dangle(
                        bkm1, bk, bl) * p_seq[k-1, bkm1] * base
                sm += jnp.where(curr == 1,
                                jnp.sum(vmap(dangle5_fn)(N4)),
                                0.0)



                cond1 = (b < n_c-1) | (last == 1)
                cond2 = (l+1 >= j)
                cond3 = (b < n_c-1) & (nexti == l+1)
                count_dangle3_term_mismatch = (cond1 & ~(cond2 | cond3))
                def dangle3_term_mismatch_fn(blp1):
                    bit2 = jnp.where(nexti > l+2, 1, 0)
                    dangle3_sm = curr_kdp[last, bit2, b+1]*em.en_3dangle(
                        bk, bl, blp1)*p_seq[l+1, blp1]*base

                    def term_mismatch_fn(bkm1):
                        return curr_kdp[last, bit2, b+1] \
                            * em.en_term_mismatch(bkm1, bk, bl, blp1) \
                            * p_seq[k-1, bkm1] * p_seq[l+1, blp1] * base

                    term_mismatch_sm = jnp.where(curr == 1,
                                                 jnp.sum(vmap(term_mismatch_fn)(N4)),
                                                 0.0)

                    return dangle3_sm + term_mismatch_sm
                sm += jnp.where(count_dangle3_term_mismatch,
                                jnp.sum(vmap(dangle3_term_mismatch_fn)(N4)),
                                0.0)

                return sm

            @jit
            def b_last_curr_fn(last, curr):
                bp_vals = vmap(b_last_curr_bp_fn, (None, None, 0))(last, curr, N6)
                return jnp.sum(bp_vals)


            get_last_curr_vals = vmap(vmap(b_last_curr_fn, (None, 0)), (0, None))

            last_curr_vals = get_last_curr_vals(jnp.arange(2), jnp.arange(2))

            curr_kdp = jnp.where(b < n_c,
                                 curr_kdp.at[:, :, b].add(last_curr_vals),
                                 curr_kdp)

            return curr_kdp, None

        kdp, _ = scan(b_fn, kdp, jnp.arange(num_c_max-1, -1, -1))
        return kdp

    @jit
    def psum_kloop(i, p_seq, dp):

        st = jnp.where(i == -1, ch_idx[-1], ch_idx[i]) # Note: `where` unecessary, but not intended so we make it explicit
        en = st + jnp.where(i == -1, num_c[-1], num_c[i]) # Note: same as above -- `where` is not necessary
        j = jnp.where(i == -1, n, right[i])

        return _psum_kloop(i, j, st, en, p_seq, dp)

    @jit
    def psum_multiloop(i, j, bi, bj, p_seq, kdp):
        lj = right[children[ch_idx[i]+num_c[i]-1]]
        fi = children[ch_idx[i]]

        bit1 = jnp.where(lj < j-1, 1, 0)
        bit2 = jnp.where(i+1 < fi, 1, 0)
        sm = kdp[bit1, bit2, 0]

        bit3 = jnp.where(i+2 < fi, 1, 0)
        @jit
        def bip1_fn(bip1):
            return kdp[bit1, bit3, 0] * \
                em.en_3dangle_inner(bi, bip1, bj) * p_seq[i+1, bip1]

        sm += jnp.where(bit2,
                        jnp.sum(vmap(bip1_fn)(N4)),
                        0.0)


        bit4 = jnp.where(lj < j-2, 1, 0)
        @jit
        def bjm1_fn(bjm1):
            bjm1_sm = kdp[bit4, bit2, 0] * \
                      em.en_5dangle_inner(bi, bjm1, bj) * p_seq[j-1, bjm1]

            @jit
            def bjm1_bip1_fn(bip1):
                return kdp[bit4, bit3, 0] * em.en_term_mismatch_inner(
                    bi, bip1, bjm1, bj) * p_seq[i+1, bip1] * p_seq[j-1, bjm1]

            bip1_sm = jnp.where(bit2,
                                jnp.sum(vmap(bjm1_bip1_fn)(N4)),
                                0.0)
            return bjm1_sm + bip1_sm

        sm += jnp.where(bit1,
                        jnp.sum(vmap(bjm1_fn)(N4)),
                        0.0)

        sm *= em.en_multi_closing(bi, bj)
        return sm


    @jit
    def bp_fn(p_seq, dp, kdp, i, j, bp):
        bi = bp[0]
        bj = bp[1]
        boltz = jnp.where(num_c[i] == 0,
                          psum_hairpin(bi, bj, i, j, p_seq),
                          jnp.where(num_c[i] == 1,
                                    psum_twoloop(i, j, bi, bj, p_seq, dp),
                                    psum_multiloop(i, j, bi, bj, p_seq, kdp)))
        return boltz


    def seq_partition(p_seq):
        dp = jnp.zeros((NBPS, n), dtype=f64)

        @jit
        def fill_dp(carry_dp, i):
            j = right[i]

            # Note: the `where` is unnecessary because if the condition isn't met, `kdp` isn't used
            # kdp = jnp.where(num_c[i] > 1, psum_kloop(i), jnp.zeros((2, 2, n+1)))
            kdp = psum_kloop(i, p_seq, carry_dp)

            bp_vals = vmap(bp_fn, (None, None, None, None, None, 0))(p_seq, carry_dp, kdp, i, j, bp_bases)
            carry_dp = carry_dp.at[:, i].set(bp_vals)
            return carry_dp, None

        fin_dp, _ = scan(fill_dp, dp, order)

        # External loop
        # kdp = _psum_kloop(i=-1, j=n, st=0, en=n_external_c, p_seq=p_seq, dp=fin_dp)
        kdp = psum_kloop(-1, p_seq, fin_dp)
        bit_last = jnp.where(right[external_c[-1]]+1 < n, 1, 0)
        bit_first = jnp.where(external_c[0] > 0, 1, 0)
        boltz = jnp.where(# n_external_c > 0,
            num_c[-1] > 0,
            kdp[bit_last, bit_first, 0],
            1.0)
        # return boltz, kdp, fin_dp
        return boltz

    return seq_partition


class TestSeqPartitionFunction(unittest.TestCase):


    def _test_dummy_ryan(self):
        em = energy.JaxNNModel("misc/rna_turner2004.par")
        # db = '.(....).(...)'
        db = "..((((((((.....))))((((.....)))))))).."

        n = len(db)
        # seq = "UAAUGAUUGUGCC"
        seq = "AAGGGGGGGGAAAAACCCCGGGGAAAAACCCCCCCCAA"
        p_seq = jnp.array(seq_to_one_hot(seq))

        seq_fn = get_seq_partition_fn(em, db)
        boltz_calc, fin_kdp, fin_dp = seq_fn(p_seq)
        # boltz_calc = seq_fn(p_seq)

        from jax_rnafold.d1.seq_reference import seq_partition
        other_boltz_ref, ref_kdp, ref_dp = seq_partition(p_seq, db, em)
        # other_boltz_ref = seq_partition(p_seq, db, em)

        pdb.set_trace()

        self.assertAlmostEqual(1.0, 1.0, places=10)

    def _test_dummy_max(self):
        em = energy.JaxNNModel("misc/rna_turner2004.par")
        # db = '.(....).(...)'
        db = "..((((((((.....))))((((.....)))))))).."

        n = len(db)
        # seq = "UAAUGAUUGUGCC"
        seq = "AAGGGGGGGGAAAAACCCCGGGGAAAAACCCCCCCCAA"
        p_seq = jnp.array(seq_to_one_hot(seq))

        from jax_rnafold.d1.seq_reference import seq_partition
        other_boltz_ref, ref_kdp, ref_dp = seq_partition(p_seq, db, em)
        # other_boltz_ref = seq_partition(p_seq, db, em)
        pdb.set_trace()
        print(f"Max's reference boltz: {other_boltz_ref}")


        self.assertAlmostEqual(1.0, 1.0, places=10)

    def _test_dummy(self):
        em = energy.JaxNNModel("misc/rna_turner2004.par")
        # db = '.(....).'
        # db = '.(....).(...)'
        # db = '..(...)(...)(....)..'
        db = '(..(...)(...)..)'
        seq_fn = get_seq_partition_fn(em, db)

        n = len(db)
        # seq = "AGUGGUUUCC"
        # seq = "UAAUGAUUGUGCC"
        seq = "CAAGAAACGAAACAAG"
        # seq = "GAGAAACGAAACGAAAACAC"
        p_seq = jnp.array(seq_to_one_hot(seq))
        # p_seq = random_pseq(n)
        # p_seq = jnp.array(p_seq)

        # boltz_calc = seq_fn(p_seq)
        boltz_calc, fin_kdp, fin_dp = seq_fn(p_seq)
        print(f"Our Seq PF: {boltz_calc}")

        boltz_ref = energy.calculate(seq, db, em)
        print(f"Energy calculator boltz: {boltz_ref}")

        from jax_rnafold.d1.seq_reference import seq_partition
        # other_boltz_ref = seq_partition(p_seq, db, em)
        other_boltz_ref, ref_kdp, ref_dp = seq_partition(p_seq, db, em)
        print(f"Max's reference boltz: {other_boltz_ref}")

        self.assertAlmostEqual(boltz_calc, boltz_ref, places=10)

    def test_vienna(self):
        em = energy.JaxNNModel("misc/rna_turner2004.par")
        self.fuzz_test(n=20, num_seq=16, em=em, tol_places=14, max_structs=50)

    def fuzz_test(self, n, num_seq, em, tol_places=6, max_structs=20):
        from jax_rnafold.common import vienna_rna, sampling
        from jax_rnafold.common.utils import dot_bracket_2_matching, matching_2_dot_bracket
        from jax_rnafold.common.utils import seq_to_one_hot, get_rand_seq, random_pseq
        import random
        from tqdm import tqdm

        seqs = [get_rand_seq(n) for _ in range(num_seq)]

        for seq in seqs:
            p_seq = jnp.array(seq_to_one_hot(seq))
            print(f"Sequence: {seq}")
            sampler = sampling.UniformStructureSampler()
            sampler.precomp(seq)
            n_structs = sampler.count_structures()
            if n_structs > max_structs:
                all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
            else:
                all_structs = [sampler.get_nth(i) for i in range(n_structs)]
            all_structs = [matching_2_dot_bracket(matching) for matching in all_structs]

            print(f"Found {len(all_structs)} structures")

            for db_str in tqdm(all_structs):
                seq_fn = get_seq_partition_fn(em, db_str)
                print(f"\n\tStructure: {db_str}")

                # boltz_calc, kdp, fin_dp = seq_fn(p_seq)
                boltz_calc = seq_fn(p_seq)
                print(f"\t\tOur Seq PF: {boltz_calc}")

                boltz_ref = energy.calculate(seq, db_str, em)
                print(f"\t\tEnergy calculator boltz: {boltz_ref}")

                # from jax_rnafold.d1.seq_reference import seq_partition
                # other_boltz_ref, ref_kdp, ref_dp, = seq_partition(p_seq, db_str, em)
                # other_boltz_ref = seq_partition(p_seq, db_str, em)
                # print(f"\t\tMax's reference boltz: {other_boltz_ref}")

                self.assertAlmostEqual(boltz_calc, boltz_ref, places=tol_places)




if __name__ == "__main__":
    unittest.main()
