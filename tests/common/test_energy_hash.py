import unittest
import pdb

from jax_rnafold.common import energy_hash

import jax

# Note: without float64, the jax version overflows and the two are not equal for the same seed
jax.config.update("jax_enable_x64", True)


class TestEnergyHash(unittest.TestCase):
    def test_simple(self):
        val = energy_hash.float_hash(4.0, 1, 2, 3)

        fn = energy_hash.get_jax_float_hash_fn(4.0)
        jax_val = fn(1, 2, 3)

        self.assertEqual(val, jax_val)
