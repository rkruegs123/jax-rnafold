import unittest
import pdb
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from jax import jit
import jax.numpy as jnp

from jax_rnafold.d2 import energy, reference
from jax_rnafold.d2.ss import get_ss_partition_fn
from jax_rnafold.common import brute_force, utils, vienna_rna
from jax_rnafold.common import nussinov as nus


class TestPartitionFunction(unittest.TestCase):

    def all_1_test(self, n):
        em_jax = energy.All1Model(jaxify=True)
        ss_partition_fn = get_ss_partition_fn(em_jax, seq_len=n)

        p_seq = utils.random_pseq(n)
        nuss = nus.ss_partition(p_seq, en_pair=nus.en_pair_1)
        vien = ss_partition_fn(p_seq)
        print(n, nuss, vien)
        self.assertAlmostEqual(nuss, vien, places=7)

    def brute_model_test(self, em_ref, em_jax, n):
        p_seq = utils.random_pseq(n)
        ss_partition_fn = get_ss_partition_fn(em_jax, seq_len=n)
        vien = ss_partition_fn(p_seq)
        brute = brute_force.ss_partition(p_seq, energy_fn=lambda seq, match: energy.calculate(
            seq, utils.matching_to_db(match), em_ref), hairpin=em_ref.hairpin)
        print(n, brute, vien)
        self.assertAlmostEqual(brute, vien, places=7)

    def reference_ss_test(self, em_ref, em_jax, n):
        p_seq = utils.random_pseq(n)
        ss_partition_fn = get_ss_partition_fn(em_jax, n)
        vien = ss_partition_fn(p_seq)
        max = reference.ss_partition(p_seq, em_ref)
        print(n, max, vien)
        self.assertAlmostEqual(max, vien, places=7)

    def random_test(self, n):
        em_ref = energy.RandomModel(jaxify=False)
        em_jax = energy.RandomModel(jaxify=True)

        # self.brute_model_test(em_ref, em_jax, n)
        self.reference_ss_test(em_ref, em_jax, n)


    def test_all_1_model_to_10(self):
        print("Starting test: [test_all_1_model_to_10]")
        for n in range(1, 10):
            self.all_1_test(n)

    def test_all_1_model_12(self):
        # First multiloop for HAIRPIN=3
        print("Starting test: [test_all_1_model_12]")
        self.all_1_test(12)

    def test_all_1_model_20(self):
        print("Starting test: [test_all_1_model_20]")
        self.all_1_test(20)


    def test_random_model_to_10(self):
        print("Starting test: [test_random_model_to_10]")
        for n in range(1, 10):
            self.random_test(n)

    def fuzz_test(self, n, num_seq, em_jax, em_ref, tol_places=6):
        ss_partition_fn = jit(get_ss_partition_fn(em_jax, n))

        for _ in range(num_seq):
            seq = utils.get_rand_seq(n)
            p_seq = jnp.array(utils.seq_to_one_hot(seq))

            pf_calc = ss_partition_fn(p_seq)

            pf_reference = reference.ss_partition(p_seq, em_ref)
            print(seq, pf_calc, pf_reference, onp.abs(pf_reference - pf_calc))

            self.assertAlmostEqual(pf_reference, pf_calc, places=tol_places)

    def test_nn_to_16(self):

        n_seq = 3
        tol_places = 12
        ns = onp.arange(3, 10)

        for hairpin in range(4):

            energy_models = {
                "Nearest Neighbor": (energy.StandardNNModel(hairpin=hairpin), energy.JaxNNModel(hairpin=hairpin)),
                "All1": (energy.All1Model(jaxify=False, hairpin=hairpin), energy.All1Model(jaxify=True, hairpin=hairpin)),
                "Random": (energy.RandomModel(jaxify=False, hairpin=hairpin), energy.RandomModel(jaxify=True, hairpin=hairpin))
            }

            for n in ns:
                for em_name, (em_ref, em_jax) in energy_models.items():
                    print(f"---------- n={n}, Energy Model={em_name} ----------")
                    self.fuzz_test(n, n_seq, em_jax, em_ref, tol_places=tol_places)
