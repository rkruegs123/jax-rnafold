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
from jax.config import config
config.update("jax_enable_x64", True)

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


f64 = jnp.float64


checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)

def get_seq_struct_partition_fn(em, db):
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

    right = jnp.array(right)
    order = jnp.array(order)
    num_c = jnp.array(num_c)
    children = jnp.array(children)
    ch_idx = jnp.array(ch_idx)
    external_c = jnp.array(external_c)

    pdb.set_trace()

    ## End precompute


    @jit
    def psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq):

        up2 = j-i+1
        u = j - i - 1

        # FIXME: repeated computation should combine with `psum_hairpin_special()`
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

    # FIXME: copied from `ss.py`. Should make the jittable version a global function
    @jit
    def psum_hairpin(bi, bj, i, j, padded_p_seq):
        return psum_hairpin_not_special(bi, bj, i, j, padded_p_seq) \
            + psum_hairpin_special(bi, bj, i, j, padded_p_seq) \
            - psum_hairpin_special_correction(bi, bj, i, j, padded_p_seq)



    def psum_twoloop(i, j, bi, bj, p_seq, dp):
        k = children[ch_idx[i]]
        l = right[k]

        lup = k-i-1
        rup = j-l-1

        def psum_twoloop_bp(bp_idx):
            bp = bp_bases[bp_idx]
            bk = bp[0]
            bl = bp[1]



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
            internal_sm = jnp.sum(internal_fn_mapped(N6, N6, N6, N6))

            stack_cond = (k == i+1) & (l == j-1)
            bulge_cond = (k == i+1) | (l == j-1)
            bulge_nunpaired = jnp.where(k-i-1 > j-l-1, k-i-1, j-l-1) # FIXME: how we want to do branchless max?
            return jnp.where(stack_cond,
                             em.en_stack(bi, bj, bk, bl)*p_seq[k, bk]*p_seq[l, bl]*dp[bp_idx, k],
                             jnp.where(bulge_cond,
                                       em.en_bulge(bi, bj, bk, bl, bulge_nunpaired) * \
                                       p_seq[k, bk]*p_seq[l, bl]*dp[bp_idx, k],
                                       internal_sm))


        bp_vals = vmap(psum_twoloop_bp)(N6)
        return jnp.sum(bp_vals)

    @jit
    def psum_kloop(i):
        st = ch_idx[i]
        en = st + num_c[i]
        j = right[i]

        left = jnp.where(i != -1, children, external_c) # FIXME: will these always be the same size?

        n = num_c[i]
        # n = en-st

        kdp = jnp.zeros((2, 2, n+1))
        kdp = kdp.at[:, :, n].set(1)

        @jit
        def branch(idx):
            left_i = left[st+idx]
            return left_i, right[left_i]

        @jit
        def next_i(idx):
            return jnp.where(idx == n-1, j, branch(idx+1)[0])


        # Note: order of b matters, so we scan
        @jit
        def b_fn(curr_kdp, b):
            b_i, b_j = branch(b)
            nexti = next_i(b)

            # Return the value to be *added* to the current entry at [last, curr, b]
            def b_last_curr_fn(last, curr):
                return 1.0 # FIXME: implement
            get_last_curr_vals = vmap(vmap(b_last_curr_fn, (None, 0)), (0, None))

            last_curr_vals = get_last_curr_vals(jnp.arange(2), jnp.arange(2))

            curr_kdp = curr_kdp.at[b, :, :].add(last_curr_vals)
            return curr_kdp, None

        kdp, _ = scan(b_fn, kdp, jnp.arange(n-1, -1, -1))
        return kdp



    def bp_fn(p_seq, dp, i, j, bp):
        bi = bp[0]
        bj = bp[1]
        boltz = jnp.where(num_c[i] == 0,
                          psum_hairpin(bi, bj, i, j, p_seq),
                          jnp.where(num_c[i] == 1, psum_twoloop(i, j, bi, bj, p_seq, dp), 0.0))
        return boltz


    def seq_partition(p_seq):
        dp = jnp.zeros((NBPS, n), dtype=f64)

        def fill_dp(carry_dp, i):
            j = right[i]

            # Note: the `where` is unnecessary because if the condition isn't met, `kdp` isn't used
            kdp = jnp.where(num_c[i] > 1, psum_kloop(i), jnp.zeros((2, 2, n+1)))

            bp_vals = vmap(bp_fn, (None, None, None, None, 0))(p_seq, carry_dp, i, j, bp_bases)
            carry_dp = carry_dp.at[:, i].set(bp_vals)
            return carry_dp, None

        fin_dp, _ = scan(fill_dp, dp, order)

        return 0.0

    return seq_partition


class TestSeqPartitionFunction(unittest.TestCase):
    def test_dummy(self):
        em = energy.JaxNNModel()
        db = '((...))'
        seq_fn = get_seq_struct_partition_fn(em, db)

        n = len(db)
        p_seq = random_pseq(n)
        p_seq = jnp.array(p_seq)

        seq_fn(p_seq)
        self.assertAlmostEqual(1.0, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
