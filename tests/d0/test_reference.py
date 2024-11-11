import unittest
from tqdm import tqdm
import numpy as onp

from jax_rnafold.common import brute_force, sampling, utils, vienna_rna
from jax_rnafold.d0 import reference, energy

class TestSSPartitionFunction(unittest.TestCase):

    def fuzz_test_brute_force(self, n, n_seqs, em, tol_places=5):
        for _ in tqdm(range(n_seqs)):
            seq = utils.get_rand_seq(n)
            p_seq = utils.seq_to_one_hot(seq)
            print(f"Sequence: {seq}")

            q = reference.ss_partition(p_seq, em)

            ref_q = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
                seq, utils.matching_to_db(match), em), hairpin=em.hairpin)
            print(f"\tCalculated SS PF: {q}")
            print(f"\tReference SS PF: {ref_q}")
            print(f"\tDifference: {onp.abs(ref_q - q)}")

            self.assertAlmostEqual(q, ref_q, places=tol_places)

    def test_brute_force(self):
        em = energy.All1Model(jaxify=False, hairpin=0)
        # self.fuzz_test_brute_force(n=10, n_seqs=50, em=em)
        self.fuzz_test_brute_force(n=7, n_seqs=10, em=em)

    def fuzz_test_vienna(self, n, n_seqs, params_path, tol_places=5):

        em = energy.StandardNNModel(params_path=params_path)
        for _ in tqdm(range(n_seqs)):
            seq = utils.get_rand_seq(n)
            p_seq = utils.seq_to_one_hot(seq)
            print(f"Sequence: {seq}")

            q = reference.ss_partition(p_seq, em)

            vc = vienna_rna.ViennaContext(seq, utils.kelvin_to_celsius(em.temp),
                                          dangles=0, params_path=params_path)
            ref_q = vc.pf()

            print(f"\tCalculated SS PF: {q}")
            print(f"\tReference SS PF: {ref_q}")
            print(f"\tDifference: {onp.abs(ref_q - q)}")

            self.assertAlmostEqual(q, ref_q, places=tol_places)

    def test_vienna(self):
        # self.fuzz_test_vienna(n=10, n_seqs=50, params_path=utils.TURNER_2004)
        self.fuzz_test_vienna(n=7, n_seqs=10, params_path=utils.TURNER_2004)
