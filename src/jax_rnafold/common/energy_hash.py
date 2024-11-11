import jax.numpy as jnp
import jax
from jax import jit
import jax.debug

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


HASH_BASE = 97
HASH_MOD = 10007

def float_hash(seed, *args):
    p = 1
    v = (seed*7777777) % HASH_MOD
    for a in args:
        if a is None:
            continue
        v += p*a
        v %= HASH_MOD
        p *= HASH_BASE
        p %= HASH_MOD
    v = (v * 9999999) % HASH_MOD
    return v/(HASH_MOD-1)


def get_jax_float_hash_fn(seed):
    def jax_float_hash_fn(*args):
        vals_to_hash = jnp.array(args)

        p_init = 1
        v_init = (seed*7777777) % HASH_MOD

        def a_fn(acc, a):
            (p, v) = acc
            v = v + (p * a)
            v = v % HASH_MOD

            p = p * HASH_BASE
            p = p % HASH_MOD

            return (p, v), None
        (p, v), _ = jax.lax.scan(a_fn, (p_init, v_init), vals_to_hash)
        v = (v * 9999999) % HASH_MOD
        return v / (HASH_MOD-1)
    return jax_float_hash_fn
