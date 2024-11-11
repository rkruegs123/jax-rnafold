import numpy as onp
import pdb
import unittest
import time
from tqdm import tqdm
import random

import jax.numpy as jnp

from jax_rnafold.common import sampling
from jax_rnafold.common import utils

from jax_rnafold.d2 import energy
from jax_rnafold.d2.seq_pf import get_seq_partition_fn


class TestSeqPartitionFunction(unittest.TestCase):

    def test_invalid_regression(self):
        tests = [("GGGCCAUUUUAGCUUUUCUUUUUUUGCAUUGGGCACC", "(((((((....((............))..))))).))")]
        tol_places = 10

        for seq, db_str in tests:
            print(f"\n\tSeq: {seq}")
            print(f"\n\tStructure: {db_str}")

            p_seq = jnp.array(utils.seq_to_one_hot(seq))

            em = energy.JaxNNModel()
            seq_partition_fn = get_seq_partition_fn(em, db_str)

            seq_pf = seq_partition_fn(p_seq)
            print(f"\t\tOur Seq PF: {seq_pf}")

            self.assertAlmostEqual(seq_pf, 0.0, places=tol_places)

    def fuzz_test(self, n, num_seq, em, max_structs=20, tol_places=6):

        seqs = [utils.get_rand_seq(n) for _ in range(num_seq)]

        failed_cases = list()
        n_passed = 0

        for seq in seqs:
            p_seq = jnp.array(utils.seq_to_one_hot(seq))

            print(f"Sequence: {seq}")
            sampler = sampling.UniformStructureSampler()
            sampler.precomp(seq)
            n_structs = sampler.count_structures()
            if n_structs > max_structs:
                all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
            else:
                all_structs = [sampler.get_nth(i) for i in range(n_structs)]
            all_structs = [utils.matching_to_db(matching) for matching in all_structs]

            print(f"Found {len(all_structs)} structures")

            for db_str in tqdm(all_structs):
                seq_partition_fn = get_seq_partition_fn(em, db_str)

                print(f"\n\tStructure: {db_str}")

                reference_seq_pf = energy.calculate(seq, db_str, em)
                print(f"\t\tReference Seq PF: {reference_seq_pf}")

                seq_pf = seq_partition_fn(p_seq)
                print(f"\t\tOur Seq PF: {seq_pf}")

                print(f"\t\tDifference: {onp.abs(seq_pf - reference_seq_pf)}")

                self.assertAlmostEqual(seq_pf, reference_seq_pf, places=tol_places)

    def test_reference(self):
        em = energy.JaxNNModel()
        n_seq = 5
        self.fuzz_test(n=20, num_seq=n_seq, em=em, tol_places=12, max_structs=50)
