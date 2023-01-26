import numpy as onp
import pdb
import functools
import unittest
import time
from pathlib import Path
import datetime
import pickle

import jax
import optax
from jax import vmap, jit, grad, value_and_grad
from jax.tree_util import Partial
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

import energy
from checkpoint import checkpoint_scan
from utils import bp_bases, HAIRPIN, N4, INVALID_BASE, RNA_ALPHA
from utils import SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_LENS, \
    SPECIAL_HAIRPIN_IDXS, N_SPECIAL_HAIRPINS, SPECIAL_HAIRPIN_START_POS
from utils import matching_to_db
from utils import MAX_PRECOMPUTE
import brute_force
import nussinov as nus

import dp_discrete
from utils import get_rand_seq, seq_to_one_hot
from ss import get_ss_partition_fn
from seq_pf import get_seq_partition_fn
import vienna_rna


def design_seq_for_struct(db_str,
                          n_iter=50, lr=0.1, optimizer="rms-prop",
                          print_every=1):
    n = len(db_str)
    em = energy.JaxNNModel()

    seq_pf_fn = jit(get_seq_partition_fn(em, db_str))
    ss_pf_fn = jit(get_ss_partition_fn(em, n))

    def neg_log_prob_fn(params, key, temp):
        curr_logits = params['seq_logits']

        p_seq = jax.nn.softmax(curr_logits)
        # gumbel_weights = jax.random.gumbel(key, curr_logits.shape)
        # p_seq = jax.nn.softmax((curr_logits  + gumbel_weights) / temp)

        seq_pf = seq_pf_fn(p_seq)
        ss_pf = ss_pf_fn(p_seq)
        return -jnp.log(seq_pf / ss_pf)
    log_prob_grad = value_and_grad(neg_log_prob_fn)
    log_prob_grad = jit(log_prob_grad)


    seq_logits = onp.full((n, 4), 5)
    seq_logits = jnp.array(seq_logits, dtype=jnp.float64)


    if optimizer == "rms-prop":
        optimizer = optax.rmsprop(learning_rate=lr)
    else:
        raise RuntimeError(f"Invalid choice of optimizer: {optimizer}")
    params = {'seq_logits': seq_logits}

    key = jax.random.PRNGKey(0)
    iter_keys = jax.random.split(key, n_iter)
    iter_temps = jnp.linspace(10, 0.1, n_iter)
    opt_state = optimizer.init(params)

    all_times = list()
    iter_params = [params] # will have n_iter+1 elements
    iter_losses = list() # will have n_iter+1 elements
    iter_grads = list() # will have n_iter elements -- not worth it to get the grad of the final params
    for i, i_key, i_temp in zip(range(n_iter), iter_keys, iter_temps):
    # for i in range(n_iter):
        start = time.time()

        loss, _grad = log_prob_grad(params, i_key, i_temp)
        updates, opt_state = optimizer.update(_grad, opt_state)
        params = optax.apply_updates(params, updates)

        iter_params.append(params)
        iter_losses.append(loss)
        iter_grads.append(_grad)

        end = time.time()
        iter_time = end - start
        all_times.append(end - start)

        if i % print_every == 0:
            # print(f"{i}: {loss}    (temp={temp})")
            print(f"{i}: {loss}")
        if i % 10 == 0:
            curr_pr_seq = jax.nn.softmax(params['seq_logits'])
            curr_maxs = jnp.argmax(curr_pr_seq, axis=1)
            curr_seq = ''.join([RNA_ALPHA[idx] for idx in curr_maxs])
            print(f"Current argmax sequence: {curr_seq}")

    final_loss, _ = log_prob_grad(params, None, None) # Won't work with gumbel trick turned on
    iter_losses.append(final_loss)

    return params, all_times, iter_params, iter_losses, iter_grads



def get_structs_within_length(n):
    bpath = Path("../misc/eterna100_vienna1.txt")

    # parse it line-by-line because whitespace prevents reading it as a dataframe easily
    f = open(bpath, 'r')
    lines = f.readlines()
    f.close()

    db_chars = set(['(', '.', ')'])
    structs_within_threshold = list()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        elts = line.split()
        # print(elts)
        structs = [seq for seq in elts if set(seq).issubset(db_chars) and len(seq) > 1]
        if not len(structs) == 1:
            pdb.set_trace()
        struct = structs[0]


        if len(struct) <= n:
            pdb.set_trace()
            print(line)
            print(f"{i}: {struct}")
            structs_within_threshold.append(struct)
    n_found = len(structs_within_threshold)
    print(f"Found {n_found} structs within threshold")
    return structs_within_threshold



def run_all(optimizer="rms-prop", lr=0.1, n_iter=200, data_basedir=Path("data/")):
    # Eterna100-V1
    structs_within_50 = {
        "Simple-Hairpin": "(((((......)))))",
        "Prion-Pseudoknot": "((((((.((((....))))))).)))..........",
        "G-C-Placement": "((((...)))).",
        "Frog-Foot": "..........((((....))))((((....))))((((...))))",
        "InfoRNA-test-16": "((((((.((((((((....))))).)).).))))))",
        "Small-and-Easy-6": "(((((.....))..((.........)))))",
        "InfoRNA-bulge-test-9": "(((((((.(.(.(.(((((((....)))))))))))))))))",
        "Shortie-4": "((....)).((....))",
        "stickshift": "..((((((((.....)).))))))..",
        "Corner-bulge-training": ".(((((((((((...)))))....)))))).",
        "Worm-1": ".......(.(.(.(.(.((.((.(....).)).)).).).).).)",
        "Tripod5": "..((((((((.....))))((((.....))))))))..",
        "Shortie-6": "((....)).((....)).((....)).((....))",
        "Misfolded-Aptamer": "((((......(((((...))).((....)).........)).....))))",
        "multilooping-fun": "((.(..(.(....).(....).)..).(....).))",
        "Branching-Loop": ".(((((........)((((....))))..)))).......",
        "Bug-38": ".((.((.((..((.(...).)).))..))....)).",
        "Zigzag-Semicircle": "....((((((((.(....)).).).)))))...."
    }

    structs_within_50_v2_incompatible = [
        "multilooping-fun", "Zigzag-Semicircle"
    ]

    params_str = ""
    params_str += "learning_rate: {lr}\n"
    params_str += "optimizer: {optimizer}\n"
    params_str += "num iterations: {n_iter}\n"

    for name, db_str in structs_within_50.items():
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f"{name}_{timestamp}"
        run_dir = data_basedir / run_name
        run_dir.mkdir(parents=False, exist_ok=False)

        with open(run_dir / "params.txt", "w+") as f:
            f.write(params_str)

        print(f"Starting optimization for {db_str}...")
        opt_params, iter_times, iter_params, iter_losses, iter_grads = design_seq_for_struct(db_str, n_iter=n_iter, lr=lr, optimizer=optimizer)

        with open(run_dir / "iter_seqs.pkl", "wb") as handle:
            pickle.dump(iter_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(run_dir / "iter_losses.pkl", "wb") as handle:
            pickle.dump(iter_losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(run_dir / "iter_grads.pkl", "wb") as handle:
            pickle.dump(iter_grads, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(run_dir / "opt_params.pkl", "wb") as handle:
            pickle.dump(opt_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(run_dir / "iter_times.pkl", "wb") as handle:
            pickle.dump(iter_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

        opt_pr_seq = jax.nn.softmax(opt_params['seq_logits'])
        opt_maxs = jnp.argmax(opt_pr_seq, axis=1)
        opt_seq = ''.join([RNA_ALPHA[idx] for idx in opt_maxs])
        print(f"Final argmax sequence: {opt_seq}")

        with open(run_dir / "opt_argmax_seq.txt", "w+") as f:
            f.write(opt_seq)

        final_loss = iter_losses[-1]
        final_neg_exp_loss = -jnp.exp(final_loss)
        final_vienna_prob = vienna_rna.vienna_prob(opt_seq, db_str)
        final_loss_str = f"final loss: {final_loss}\n"
        final_loss_str += f"final neg exp loss: {final_neg_exp_loss}\n"
        final_loss_str += f"final vienna prob: {final_vienna_prob}\n"
        with open(run_dir / "final_loss.txt", "w+") as f:
            f.write(final_loss_str)


if __name__ == "__main__":
    run_all()
    pdb.set_trace()


    # get_structs_within_length(50)
    # pdb.set_trace()

    """
    test_struct = "..((((((((.....))))((((.....)))))))).." # tripod
    opt_params, all_times, _, _, _ = design_seq_for_struct(test_struct, n_iter=100)
    opt_pr_seq = jax.nn.softmax(opt_params['seq_logits'])
    maxs = jnp.argmax(opt_pr_seq, axis=1)
    nucs = [RNA_ALPHA[idx] for idx in maxs]
    fin_seq = ''.join(nucs)
    print(f"Final argmax sequence: {fin_seq}")
    """




