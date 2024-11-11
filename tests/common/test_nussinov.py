import unittest
import numpy as np

from jax_rnafold.common import brute_force, nussinov, utils


class TestNussinov(unittest.TestCase):
    def test_simple(self):
        n = 8
        p_seq = utils.random_pseq(n)
        hairpin = utils.DEFAULT_HAIRPIN

        nus_val = nussinov.ss_partition(p_seq, hairpin=hairpin)
        brute_val = brute_force.ss_partition(p_seq, energy_fn=nussinov.energy, hairpin=hairpin)

        diff = np.abs(nus_val-brute_val)
        rdiff = diff / min(nus_val, brute_val)

        tol_places = 14
        self.assertAlmostEqual(rdiff, 0.0, places=tol_places)
