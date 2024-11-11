from typing import Any, Callable, Sequence
import pdb
import pandas as pd
import argparse

import jax
jax.config.update("jax_enable_x64", True)
from jax import lax, random, jit, numpy as jnp
from flax.core import freeze, unfreeze
from flax import linen as nn
import optax
import jax.debug

from jax_rnafold.common.utils import ETERNA_185x, TURNER_2004
from jax_rnafold.common.utils import RNA_ALPHA
from jax_rnafold.common import utils, vienna_rna
from jax_rnafold.d2 import energy as energy_d2
from jax_rnafold.d2 import ss as ss_d2
from jax_rnafold.d2 import seq_pf as seq_pf_d2



class MLP(nn.Module):
    features: int
    layers: int
    nts: int

    @nn.compact
    def __call__(self, x, training: bool):
        for _ in range(self.layers):
            x = nn.Dense(self.features)(x)
            x = nn.leaky_relu(x)
            # x = nn.Dropout(0.5, deterministic=not training)(x)
        x = nn.Dense(self.nts*4, use_bias=True)(x)
        x = x.reshape((self.nts, 4))
        return x

puzzles_path = utils.DATA_BASEDIR / "eterna100_vienna1.csv"
puzzles_df = pd.read_csv(puzzles_path)

def run(args):

    # Load the puzzle
    puzzle_idx = args['puzzle_idx']
    puzzle_row = puzzles_df[puzzles_df['Puzzle #'] == puzzle_idx].iloc[0]
    puzzle_name = puzzle_row['Puzzle Name']
    eterna_id = puzzle_row['Eterna ID']
    author = puzzle_row['Author']
    db_str = puzzle_row['Secondary Structure']
    n = len(db_str)

    # Construct an energy model and method(s) for computing partition function(s)
    em = energy_d2.JaxNNModel(params_path=TURNER_2004)
    seq_pf_fn = jit(seq_pf_d2.get_seq_partition_fn(em, db_str))
    ss_pf_fn = jit(ss_d2.get_ss_partition_fn(em, n))


    # Construct a neural network for overparameterization
    model = MLP(layers=args['nn_layers'], features=args['nn_features'], nts=n)
    key = random.PRNGKey(args['key'])
    key, params_key, dropout_key = random.split(key, num=3)
    rand_input = random.normal(key, (10,))
    params = model.init(params_key, rand_input, training=False) # Initialization call

    # Define a loss function
    @jit
    def loss_fn(model_params):
        logits = model.apply(model_params, rand_input, training=True, rngs={'dropout': dropout_key})
        p_seq = jax.nn.softmax(logits)
        scaled_seq_pf = seq_pf_fn(p_seq)
        scaled_ss_pf = ss_pf_fn(p_seq)
        neg_log_prob = -jnp.log(scaled_seq_pf / scaled_ss_pf)
        return neg_log_prob

    # Setup the optimizer
    learning_rate = args['lr']
    optimizer = optax.fromage(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.value_and_grad(loss_fn)


    # Perform gradient descent
    losses = []
    probs = []
    for i in range(args['n_iters']):
        loss, grads = loss_grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        res = model.apply(params, rand_input, training=False)
        sm = nn.softmax(res)
        bases = jnp.argmax(sm, axis=-1)
        rna = ''.join([RNA_ALPHA[b] for b in bases])
        vc = vienna_rna.ViennaContext(rna)
        vienna_prob = vc.prob(db_str)
        losses.append(loss)
        probs.append(vienna_prob)

        print(f"Iteration {i}:")
        print(f"- Loss: {loss}")
        print(f"- Probability: {jnp.exp(-loss)}")
        if i % 10 == 0:
            print(f"- Argmax sequence: {rna}")
            print(f"- Argmax probability: {vienna_prob}")
        print("\n")


    return losses, probs




def get_argparse():
    parser = argparse.ArgumentParser(description="Attempt to solve an Eterna puzzle")

    # General arguments
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Num. iterations of gradient descent")

    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for optimization")
    parser.add_argument('-p', '--puzzle-idx', type=int, default=15, help="Eterna puzzle index")
    parser.add_argument('-k', '--key', default=111122211, type=int, help="Random key") # Used 111122211, 8899991191, and 6969 for reported experiments.

    # Neural-network-specific
    parser.add_argument('--nn-features', type=int, default=4000,
                        help="Number of features for the MLP")
    parser.add_argument('--nn-layers', type=int, default=6,
                        help="Number of layers for the MLP")

    return parser


if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())

    losses, probs = run(args)
