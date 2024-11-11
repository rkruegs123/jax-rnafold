import unittest
import pdb
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import importlib.resources

import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp

from jax_rnafold.d0 import energy, reference
from jax_rnafold.d0.ss import get_ss_partition_fn
from jax_rnafold.common import brute_force, utils, vienna_rna


REGR_TEST_BASEDIR = importlib.resources.files("jax_rnafold.data.regression-tests")

class TestSSPartitionFunction(unittest.TestCase):

    def test_regression_pseq(self):

        basedir = Path(REGR_TEST_BASEDIR / "pseq-tests")
        assert(basedir.exists() and basedir.is_dir())

        pseq_dir = basedir / "pseq"
        assert(pseq_dir.exists() and pseq_dir.is_dir())

        # Get the lengths for which we have regression tests
        lengths = list()
        fpaths = list()
        pseq_regr_test_files = basedir.glob("*.csv")
        prefix = "pseq_regr_tests_n"
        prefix_len = len(prefix)
        for fname in pseq_regr_test_files:
            fpaths.append(fname)
            fname_stem = fname.stem
            assert(fname_stem[:prefix_len] == prefix)
            length = int(fname_stem[prefix_len:])
            lengths.append(length)

        # Test the sequences for each length
        for l_idx in tqdm(range(len(lengths))):
            length = lengths[l_idx]
            length_fpath = fpaths[l_idx]

            length_pseq_dir = pseq_dir / f"n{length}"
            assert(length_pseq_dir.exists() and length_pseq_dir.is_dir())

            scale = -length * 3/2
            em_jax = energy.JaxNNModel()
            ss_fn = get_ss_partition_fn(em_jax, length, scale=scale)
            ss_fn = jit(ss_fn)

            length_df = pd.read_csv(length_fpath)
            n_seqs = length_df.shape[0]

            for row_idx in tqdm(range(n_seqs)):
                row = length_df.iloc[row_idx]
                seq_idx = int(row.seq_idx)
                ref_log_ss = row.log_ss

                pseq_path = length_pseq_dir / f"pseq_i{seq_idx}.npy"
                pseq = jnp.load(pseq_path)

                scaled_ss_pf = ss_fn(pseq)
                log_scaled_ss_pf = jnp.log(scaled_ss_pf)
                log_unscaled_ss_pf = log_scaled_ss_pf - scale

                self.assertAlmostEqual(ref_log_ss, log_unscaled_ss_pf, places=12)

        return

    def test_regression_one_hot(self):

        basedir = Path(REGR_TEST_BASEDIR / "one-hot-tests")
        assert(basedir.exists() and basedir.is_dir())

        # Get the lengths for which we have regression tests
        lengths = list()
        fpaths = list()
        oh_regr_test_files = basedir.glob("*.csv")
        prefix = "oh_regr_tests_n"
        prefix_len = len(prefix)
        for fname in oh_regr_test_files:
            fpaths.append(fname)
            fname_stem = fname.stem
            assert(fname_stem[:prefix_len] == prefix)
            length = int(fname_stem[prefix_len:])
            lengths.append(length)

        # Test the sequences for each length
        for l_idx in tqdm(range(len(lengths))):
            length = lengths[l_idx]
            length_fpath = fpaths[l_idx]

            scale = -length * 3/2
            em_jax = energy.JaxNNModel()
            ss_fn = get_ss_partition_fn(em_jax, length, scale=scale)
            ss_fn = jit(ss_fn)

            length_df = pd.read_csv(length_fpath)
            n_seqs = length_df.shape[0]
            for row_idx in tqdm(range(n_seqs)):
                row = length_df.iloc[row_idx]
                seq = row.seq
                ref_log_ss = row.log_ss
                pseq = jnp.array(utils.seq_to_one_hot(seq))

                scaled_ss_pf = ss_fn(pseq)
                log_scaled_ss_pf = jnp.log(scaled_ss_pf)
                log_unscaled_ss_pf = log_scaled_ss_pf - scale

                self.assertAlmostEqual(ref_log_ss, log_unscaled_ss_pf, places=12)

        return


    def fuzz_test_reference(self, n, n_seq, em_ref, em_jax, tol_places=12, one_hot=False):
        ss_fn = get_ss_partition_fn(em_jax, n)
        ss_fn = jit(ss_fn)
        for i in tqdm(range(n_seq)):
            if one_hot:
                seq = utils.get_rand_seq(n)
                p_seq = utils.seq_to_one_hot(seq)
                print(f"Sequence {i}: {seq}")
            else:
                p_seq = utils.random_pseq(n)
                print(f"Rand pseq: {i}")

            q = ss_fn(p_seq)
            print(f"\tCalc: {q}")

            q_ref = reference.ss_partition(p_seq, em_ref)
            print(f"\tReference: {q_ref}")

            diff = onp.abs(q-q_ref)
            print(f"\tDifference: {diff}")

            rdiff = diff / min(q, q_ref)
            print(f"\tRel. Difference: {rdiff}")

            # self.assertAlmostEqual(q, q_ref, places=tol_places)
            self.assertAlmostEqual(rdiff, 0.0, places=tol_places)

    def fuzz_test_vienna(self, n, n_seq, tol_places=6):
        scale = -n * 3/2
        em_jax = energy.JaxNNModel()
        ss_fn = get_ss_partition_fn(em_jax, n, scale=scale)
        ss_fn = jit(ss_fn)
        for i in tqdm(range(n_seq)):

            seq = utils.get_rand_seq(n)
            p_seq = utils.seq_to_one_hot(seq)
            print(f"Sequence {i}: {seq}")

            scaled_q = ss_fn(p_seq)
            q = jnp.exp(-scale) * scaled_q

            vc = vienna_rna.ViennaContext(seq, utils.kelvin_to_celsius(em_jax.temp), dangles=0,
                                          params_path=em_jax.nn_params.params_path)
            q_ref = vc.pf()

            print(f"\tCalc: {q}")
            print(f"\tReference: {q_ref}")
            diff = onp.abs(q-q_ref)
            print(f"\tDifference: {diff}")
            rdiff = diff / min(q, q_ref)
            print(f"\tRel. Difference: {rdiff}")
            # self.assertAlmostEqual(q, q_ref, places=tol_places)
            self.assertAlmostEqual(rdiff, 0.0, places=tol_places)

    def test_fuzz_reference(self):
        # em_jax = energy.All1Model(jaxify=True)
        # em_ref = energy.All1Model(jaxify=False)

        em_jax = energy.JaxNNModel(hairpin=3)
        em_ref = energy.StandardNNModel(hairpin=3)

        n_seq = 3
        self.fuzz_test_reference(15, n_seq, em_ref, em_jax, one_hot=False)

    def test_fuzz_vienna(self):
        n_seq = 3
        self.fuzz_test_vienna(40, n_seq, tol_places=3)

    def _test_regression(self, seqs, tol_places, checkpoint_every=10):
        em_jax = energy.JaxNNModel()

        seqs_per_len = dict()
        for seq in seqs:
            n = len(seq)
            if n in seqs_per_len:
                seqs_per_len[n].append(seq)
            else:
                seqs_per_len[n] = [seq]

        for n, n_seqs in seqs_per_len.items():
            print(f"\n----- Sequences of length {n} -----\n")
            scale = -n * 3/2
            ss_fn = get_ss_partition_fn(em_jax, n, scale=scale, checkpoint_every=checkpoint_every)
            ss_fn = jit(ss_fn)

            for i, seq in enumerate(n_seqs):

                p_seq = utils.seq_to_one_hot(seq)
                print(f"Sequence {i} (n={n}): {seq}")

                scaled_q = ss_fn(p_seq)
                q = jnp.exp(-scale) * scaled_q
                print(f"\tCalc: {q}")


                vc = vienna_rna.ViennaContext(seq, utils.kelvin_to_celsius(em_jax.temp), dangles=0)
                q_ref = vc.pf()
                print(f"\tReference: {q_ref}")

                diff = onp.abs(q-q_ref)
                print(f"\tDifference: {diff}")

                rdiff = diff / min(q, q_ref)
                print(f"\tRel. Difference: {rdiff}")

                # self.assertAlmostEqual(q, q_ref, places=tol_places)
                self.assertAlmostEqual(rdiff, 0.0, places=tol_places)

    def test_regression_sp_hairpin(self):
        triloops = ["CAACG", "GUUAC"]
        tetraloops = ["CAACGG", "CCAAGG", "CCACGG", "CCCAGG", "CCGAGG",
                      "CCGCGG", "CCUAGG", "CCUCGG", "CUAAGG", "CUACGG",
                      "CUCAGG", "CUCCGG", "CUGCGG", "CUUAGG", "CUUCGG",
                      "CUUUGG"]
        hexaloops = ["ACAGUACU", "ACAGUGAU", "ACAGUGCU", "ACAGUGUU"]

        # seqs = list()
        seqs = triloops + tetraloops + hexaloops
        for interior in triloops + tetraloops + hexaloops:
            seqs.append("GG" + interior + "CC")

        self._test_regression(seqs, tol_places=6, checkpoint_every=None)

    def test_regression_misc(self):
        misc_seqs = ["AAGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGAAA"]
        self._test_regression(misc_seqs, tol_places=6)
