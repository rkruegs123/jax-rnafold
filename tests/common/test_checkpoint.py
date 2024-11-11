import unittest
import functools

import jax
import jax.numpy as jnp

from jax_rnafold.common import checkpoint


class TestCheckpoint(unittest.TestCase):

    def test_simple(self):

        checkpoint_every = 2

        def foo(carry, i):
            return carry+i, i*2

        # Test the default scan
        sm, summands_x2 = jax.lax.scan(foo, 0, jnp.arange(4)) # No residual
        self.assertEqual(sm, summands_x2.sum() / 2)

        sm, summands_x2 = jax.lax.scan(foo, 0, jnp.arange(5)) # Some residual
        self.assertEqual(sm, summands_x2.sum() / 2)

        # Test the checkpointing utility
        scan = functools.partial(checkpoint.checkpoint_scan,
                                 checkpoint_every=checkpoint_every)

        sm, summands_x2 = scan(foo, 0, jnp.arange(4)) # No residual
        self.assertEqual(sm, summands_x2.sum() / 2)

        sm, summands_x2 = scan(foo, 0, jnp.arange(5)) # Some residual
        self.assertEqual(sm, summands_x2.sum() / 2)
