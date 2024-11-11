import unittest
from tqdm import tqdm
import random
import numpy as onp

import jax.numpy as jnp
from jax import jit

from jax_rnafold.common import utils, protein




class TestProtein(unittest.TestCase):
    table_path = protein.HOMOSAPIENS_CFS_PATH
    cfs = protein.CodonFrequencyTable(table_path)

    def test_expected_cai_random_weights(self):

        n = 8
        # n = 10
        n_trials = 10

        for _ in tqdm(range(n_trials)):

            codon_choices = []
            codons = 4
            for i in range(n):
                prob_dist = [random.random() for _ in range(codons)]
                norml_sum = sum(prob_dist)
                prob_dist = [x/norml_sum for x in prob_dist]
                codon_freq = [random.randint(1, 10000) for _ in range(codons)]
                mx = max(codon_freq)
                ws = [freq/mx for freq in codon_freq]
                codon_choices.append(list(zip(ws, prob_dist)))

            e_cai = protein.fast_expected_cai(codon_choices)
            print(f"Fast: {e_cai}")
            brute_e_cai = protein.brute_force_expected_cai(codon_choices)
            print(f"Brute: {brute_e_cai}")
            diff = onp.abs(e_cai - brute_e_cai)
            print(f"Diff: {diff}")

            self.assertAlmostEqual(e_cai, brute_e_cai)

    def test_expected_cai_one_hot(self):
        n_aa = 10
        aa_seq = protein.get_rand_aa_seq(n_aa)

        expected_cai_fn = protein.get_expected_cai_fn(aa_seq, self.cfs, invalid_weight=0.0)
        cds = protein.random_cds(aa_seq, self.cfs)
        cds_oh = jnp.array(utils.seq_to_one_hot(''.join(cds)))

        cai = expected_cai_fn(cds_oh)
        discrete_cai = self.cfs.codon_adaptation_index(cds)

        self.assertAlmostEqual(cai, discrete_cai, places=7)

    def test_expected_cai_probabilistic(self):
        n_aa = 100

        num_aa_seqs = 10
        num_nt_seqs_per_aa_seq = 25

        for _ in tqdm(range(num_aa_seqs)):
            aa_seq = protein.get_rand_aa_seq(n_aa)

            expected_cai_fn = protein.get_expected_cai_fn(aa_seq, self.cfs, invalid_weight=0.0)
            expected_cai_fn = jit(expected_cai_fn)

            for _ in tqdm(range(num_nt_seqs_per_aa_seq)):

                p_seq = jnp.array(utils.random_pseq(n_aa * 3))

                cai = expected_cai_fn(p_seq)
                cai_alt = protein.expected_cai_for_target(p_seq, aa_seq, self.cfs)

                print(f"CAI: {cai}")
                print(f"Reference: {cai_alt}")
                diff = onp.abs(cai - cai_alt)
                print(f"Diff: {diff}")

                self.assertAlmostEqual(cai, cai_alt)
