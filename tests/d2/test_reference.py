import unittest
from tqdm import tqdm
import numpy as np
import random

from jax_rnafold.common import brute_force, sampling, utils
import jax_rnafold.common.nussinov as nus
from jax_rnafold.d2 import reference, energy


class TestPartitionFunction(unittest.TestCase):
    def _random_p_seq(self, n):
        p_seq = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            p_seq[i] = np.random.random_sample(4)
            p_seq[i] /= np.sum(p_seq[i])
        return p_seq

    def _all_1_test(self, n):
        em = energy.All1Model()
        p_seq = self._random_p_seq(n)
        nuss = nus.ss_partition(p_seq, en_pair=nus.en_pair_1)
        vien = reference.ss_partition(p_seq, em)
        print(n, nuss, vien)
        self.assertAlmostEqual(nuss, vien, places=7)

    def _brute_model_test(self, em, n):
        p_seq = self._random_p_seq(n)
        vien = reference.ss_partition(p_seq, em)
        brute = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, utils.matching_to_db(match), em), hairpin=em.hairpin)
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
        for n in range(1, 20):
            seq = utils.get_rand_seq(n)
            p_seq = utils.seq_to_one_hot(seq)
            uss = sampling.UniformStructureSampler()
            uss.precomp(seq)
            db = utils.matching_to_db(uss.get_nth(
                random.randrange(0, uss.count_structures())))
            em = energy.RandomModel()
            e_calc = energy.calculate(seq, db, em)
            e_spart = reference.seq_partition(p_seq, db, em)
            print(n, seq, db, e_calc, e_spart)
            self.assertAlmostEqual(e_calc, e_spart, places=7)

    def test_seq_brute(self):
        n_seq = 3
        for it in range(n_seq):
            for n in range(1, 11):
                p_seq = self._random_p_seq(n)
                uss = sampling.UniformStructureSampler()
                uss.precomp([None]*n)
                db = utils.matching_to_db(uss.get_nth(random.randrange(0, uss.count_structures())))
                em = energy.RandomModel()
                e_brute = brute_force.seq_partition(p_seq, db, lambda seq, db: energy.calculate(seq, db, em))
                e_spart = reference.seq_partition(p_seq, db, em)
                print(n, db, e_brute, e_spart)
                self.assertAlmostEqual(e_brute, e_spart, places=7)
