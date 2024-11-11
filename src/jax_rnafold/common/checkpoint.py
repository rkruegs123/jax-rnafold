"""Control flow functions."""

import jax
from jax import tree_util, tree
import jax.numpy as jnp

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


def _split_n_stack(x, n):
    """Splits `x` into `n` parts along axis 0 and stackes the resulting arrays."""
    return jax.tree.map(lambda y: jnp.stack(jnp.split(y, n)), x)


def _flatten_n(x, n):
    """Flattens the first `n` dimensions of `xs`"""
    return jax.tree.map(lambda y: jnp.reshape(y, (-1,) + y.shape[n:]), x)


def checkpoint_scan(f, init, xs, checkpoint_every):
    """Replicates the behavior of `jax.lax.scan` but checkpoints gradients every `checkpoint_every` steps."""

    @jax.checkpoint
    def inner_loop(_init, _xs):
        return jax.lax.scan(f, _init, _xs)

    flat_xs, _ = jax.tree_util.tree_flatten(xs)
    length = flat_xs[0].shape[0]
    outer_iterations, residual = divmod(length, checkpoint_every)

    # Index to split the evenly-divisible portion and the residual portion
    residual_idx = length - residual

    # First, get the contribution of the evenly-divisible portion
    reshaped_xs = _split_n_stack(xs[:residual_idx], outer_iterations)
    final, result = jax.lax.scan(inner_loop, init, reshaped_xs)

    # Extend the result to include the contribution from the residual
    if residual:
        final, result_res = inner_loop(final, xs[residual_idx:])
        if result is not None:
            result = jnp.concatenate([_flatten_n(result, 2), result_res])

    return final, _flatten_n(result, 2)
