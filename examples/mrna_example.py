import numpy as onp
import pdb
import time
from pathlib import Path
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from Bio import SeqIO
import scipy
from pprint import pprint
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64", True)
import optax
from jax import jit, grad, value_and_grad
import jax.numpy as jnp
from flax import linen as nn

from jax_rnafold.common import vienna_rna, utils
import jax_rnafold.common.protein as protein

from jax_rnafold.d0 import energy as energy_d0
from jax_rnafold.d0 import ss as ss_d0



class MLP(nn.Module):
    features: int
    layers: int
    nts: int

    @nn.compact
    def __call__(self, x, training: bool):
        for _ in range(self.layers):
            x = nn.Dense(self.features)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.nts*4, use_bias=True)(x)
        x = x.reshape((self.nts, 4))
        return x


class MrnaDesignProblem:
    def __init__(self, args):
        self.args = args
        self.dangles = 0

        self.cai_threshold = args['cai_threshold']
        self.cai_scale = args['cai_scale']
        if self.cai_threshold <= 0 or self.cai_threshold >= 1:
            raise RuntimeError(f"CAI threshold must be in (0, 1)")
        if self.cai_scale < 0 or self.cai_scale >= 1:
            raise RuntimeError(f"CAI scale must be in [0, 1)")

        # Load the amino acid sequence
        records = list(SeqIO.parse(args['fasta_path'], "fasta"))
        assert(len(records) == 1)
        self.aa_seq = str(records[0].seq)
        assert(all([protein.is_valid_aa_letter(aa) for aa in self.aa_seq]))
        self.n_aa = len(self.aa_seq)
        self.n_bases_coding = self.n_aa * 3
        self.n_bases = self.n_bases_coding

        # Load the CFS table
        self.cfs = protein.CodonFrequencyTable(args['cfs_table_path'])

        # Create the run directory
        output_basedir = Path(args['output_dir'])
        if not output_basedir.exists():
            raise RuntimeError(f"No output directory exists at location: {self.output_basedir}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = args['run_name']
        if run_name is None:
            run_name = f"mrna_design_{timestamp}"
        self.run_dir = output_basedir / run_name
        self.run_dir.mkdir(parents=False, exist_ok=False)
        print(utils.bcolors.WARNING + f"Created directory at location: {self.run_dir}" + utils.bcolors.ENDC)

        params_str = ""
        params_str += f"target sequence: {self.aa_seq}\n"
        for k, v in args.items():
            params_str += f"{k}: {v}\n"
        with open(self.run_dir / "params.txt", "w+") as f:
            f.write(params_str)


        # Construct components of loss function: energy model, PF function, and CAI function
        self.scale = args['scale']
        self.params_path = args['params_path']
        self.loss_coeff = args['loss_coeff']
        if self.scale is None:
            self.scale = -self.n_bases * 3/2
        em = energy_d0.JaxNNModel(params_path=self.params_path)
        ss_pf_fn = ss_d0.get_ss_partition_fn(
            em, self.n_bases, scale=self.scale, checkpoint_every=args['checkpoint_every']
        )
        self.ss_pf_fn = jit(ss_pf_fn)

        self.unscaled_expected_cai_fn = protein.get_expected_cai_fn(
            self.aa_seq, self.cfs, restrict=False)
        n_iters = args['n_iters']

        self.thresholds = jnp.array([args['valid_seq_pr_thresh']]*n_iters)
        self.valid_seq_pr_fn = protein.get_valid_seq_pr_fn(self.aa_seq, self.cfs)

        # Set the initial logits
        start_nt_path = args['start_nt_path']
        if start_nt_path is not None:
            records = list(SeqIO.parse(start_nt_path, "fasta"))
            assert(len(records) == 1)
            nt_seq = str(records[0].seq)
            init_logits = jnp.array(utils.seq_to_one_hot(nt_seq)) * 10 + 10
        else:
            # default -- uniform codon logits
            init_logits = jnp.array(self.cfs.get_uniform_codon_logits(self.aa_seq)) * 10 # note: fixed coeff
        assert(init_logits.shape[0] == self.n_bases)
        self.init_logits = init_logits



    def scale_cai(self, cai):
        return jnp.where(
            cai < self.cai_threshold, cai,
            self.cai_threshold + self.cai_scale * (cai - self.cai_threshold))

    def scale_valid_seq_pr(self, prob, i):
        threshold = self.thresholds[i]

        poly_coeff = 50
        linear_coeff = 0.025
        return jnp.where(prob >= threshold,
                         threshold + linear_coeff*(prob - threshold),
                         -poly_coeff*(prob-threshold)**2 + threshold)

    def calc_aup(self, nuc_seq):
        vc = vienna_rna.ViennaContext(nuc_seq, dangles=self.dangles, params_path=self.params_path)
        return vc.calc_aup()

    def calc_mfe(self, nuc_seq):
        vc = vienna_rna.ViennaContext(nuc_seq, dangles=self.dangles, params_path=self.params_path)
        return vc.mfe()

    def calc_efe(self, nuc_seq):
        vc = vienna_rna.ViennaContext(nuc_seq, dangles=self.dangles, params_path=self.params_path)
        return vc.efe()

    def loss_fn(self, logits, i):
        p_seq = jax.nn.softmax(logits)

        scaled_ss_pf = self.ss_pf_fn(p_seq)
        unscaled_ss_pf = scaled_ss_pf * jnp.exp(-self.scale)
        log_scaled_ss_pf = jnp.log(scaled_ss_pf)
        log_unscaled_ss_pf = log_scaled_ss_pf - self.scale # equal to jnp.log(unscaled_ss_pf)

        # Construct loss value to maximize EFE
        loss_val = -log_unscaled_ss_pf * self.loss_coeff

        pseq_coding = p_seq[:self.n_bases_coding]
        unscaled_expected_cai = self.unscaled_expected_cai_fn(pseq_coding)
        scaled_expected_cai = self.scale_cai(unscaled_expected_cai)

        unscaled_valid_seq_pr = self.valid_seq_pr_fn(pseq_coding)
        scaled_valid_seq_pr = self.scale_valid_seq_pr(unscaled_valid_seq_pr, i)

        if self.args['no_cai']:
            loss = loss_val * scaled_valid_seq_pr
        else:
            loss = loss_val * scaled_expected_cai * scaled_valid_seq_pr

        aux = (scaled_ss_pf, log_unscaled_ss_pf, scaled_expected_cai, unscaled_expected_cai,
               scaled_valid_seq_pr, unscaled_valid_seq_pr)
        return loss, aux



def run(dp : MrnaDesignProblem):
    print(utils.bcolors.OKBLUE + f"Running optimization..." + utils.bcolors.ENDC)

    log_dir = dp.run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)
    print(utils.bcolors.WARNING + f"Created directory at location: {log_dir}" + utils.bcolors.ENDC)

    pseq_dir = dp.run_dir / "pseq"
    pseq_dir.mkdir(parents=False, exist_ok=False)
    print(utils.bcolors.WARNING + f"Created directory at location: {pseq_dir}" + utils.bcolors.ENDC)


    key = jax.random.PRNGKey(dp.args['key_seed'])


    lr = dp.args['lr']

    model = MLP(layers=dp.args['nn_layers'], features=dp.args['nn_features'], nts=dp.n_bases)

    input_seed_size = 10
    key, params_key, init_key = jax.random.split(key, 3)
    example_input_seed = jax.random.normal(init_key, (input_seed_size,))
    params = model.init(params_key, example_input_seed, training=False)

    # Note: for now, we use the same seed(s) at each training epoch
    num_training_seeds = 1
    key, training_seeds_key = jax.random.split(key)
    training_seed_keys = jax.random.split(training_seeds_key, num_training_seeds)
    seeds_list = [jax.random.normal(seed_key, (input_seed_size,)) for seed_key in training_seed_keys]

    pretrain_loss_path = log_dir / "pretrain_loss.txt"
    pretrain_nn_params_path = log_dir / "pretrain_nn_params.txt"

    ## Pretrain the network to output the initial logits
    target_logits = dp.init_logits

    @jit
    def pretrain_loss_fn(model_params):
        logits = model.apply(model_params, seeds_list[0], training=True)
        return jnp.mean((logits - target_logits)**2)

    pretrain_grad_fn = value_and_grad(pretrain_loss_fn)
    pretrain_grad_fn = jit(pretrain_grad_fn)
    pretrain_optimizer = utils.get_optimizer(dp.args['pretrain_optimizer'], dp.args['pretrain_lr'])
    opt_state = pretrain_optimizer.init(params)
    for i in range(dp.args['num_pretrain_iter']):
        loss, grads = pretrain_grad_fn(params)
        print(f"- Pretrain Iteration: {i}")
        print(f"- Loss: {loss}")
        with open(pretrain_loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(pretrain_nn_params_path, "a") as f:
            f.write(f"{params}\n")
        updates, opt_state = pretrain_optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


    @jit
    def loss_fn(model_params, i):
        seed = seeds_list[0]
        logits = model.apply(model_params, seed, training=True)
        p_seq = jax.nn.softmax(logits)
        loss, aux = dp.loss_fn(logits, i)
        scaled_ss_pf, log_unscaled_ss_pf = aux[:2]
        scaled_expected_cai, unscaled_expected_cai = aux[2:4]
        scaled_valid_seq_pr, unscaled_valid_seq_pr = aux[4:6]
        return loss, (log_unscaled_ss_pf, scaled_expected_cai, unscaled_expected_cai, scaled_valid_seq_pr, unscaled_valid_seq_pr, logits, p_seq)

    optimizer = utils.get_optimizer(dp.args['optimizer'], lr)
    opt_state = optimizer.init(params)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    loss_path = log_dir / "loss.txt"
    min_loss_path = log_dir / "min_loss.txt"
    scaled_expected_cai_path = log_dir / "scaled_expected_cai.txt"
    unscaled_expected_cai_path = log_dir / "unscaled_expected_cai.txt"
    scaled_valid_seq_pr_path = log_dir / "scaled_valid_seq_pr.txt"
    unscaled_valid_seq_pr_path = log_dir / "unscaled_valid_seq_pr.txt"
    log_ss_pf_path = log_dir / "log_ss_pf.txt"
    nuc_seq_path = log_dir / "nuc_seq.txt"
    aa_seq_path = log_dir / "aa_seq.txt"
    nuc_seq_diff_path = log_dir / "nuc_seq_diff.txt"
    aa_seq_diff_path = log_dir / "aa_seq_diff.txt"
    aup_path = log_dir / "aup.txt"
    mfe_path = log_dir / "mfe.txt"
    efe_path = log_dir / "efe.txt"
    times_path = log_dir / "times.txt"
    argmax_cai_path = log_dir / "argmax_cai.txt"
    loss_rel_change_path = log_dir / "loss_rel_change.txt"
    valid_seq_threshold_path = log_dir / "valid_seq_threshold.txt"


    def get_seqs(logits):
        curr_pr_seq = jax.nn.softmax(logits)
        curr_maxs = jnp.argmax(curr_pr_seq, axis=1)
        curr_seq = ''.join([utils.RNA_ALPHA[idx] for idx in curr_maxs])
        curr_aa_seq = dp.cfs.nuc_seq_to_aa_seq(curr_seq[:dp.n_bases_coding])
        return curr_seq, curr_aa_seq, curr_pr_seq, curr_maxs

    all_rel_changes = list()
    all_losses = list()
    curr_nt_seq, curr_aa_seq, _, _ = get_seqs(dp.init_logits)
    for i in range(dp.args['n_iters']):
        start = time.time()
        (loss, aux), grads = grad_fn(params, i)
        (log_unscaled_ss_pf, scaled_expected_cai, unscaled_expected_cai, scaled_valid_seq_pr, unscaled_valid_seq_pr, logits, p_seq) = aux
        end = time.time()
        iter_time = end - start

        all_losses.append(loss)
        if i == 0:
            rel_change = 0.0
        else:
            rel_change = utils.relative_change(all_losses[-2], all_losses[-1])
        all_rel_changes.append(rel_change)

        if i % dp.args['log_every'] == 0:
            print(f"\n---------- Iteration {i} ----------")
            print(f"- Loss: {loss}")

            with open(valid_seq_threshold_path, "a") as f:
                f.write(f"{dp.thresholds[i]}\n")
            with open(loss_rel_change_path, "a") as f:
                f.write(f"{rel_change}\n")
            with open(loss_path, "a") as f:
                f.write(f"{loss}\n")
            with open(min_loss_path, "a") as f:
                f.write(f"{onp.min(all_losses)}\n")
            with open(scaled_expected_cai_path, "a") as f:
                f.write(f"{scaled_expected_cai}\n")
            with open(unscaled_expected_cai_path, "a") as f:
                f.write(f"{unscaled_expected_cai}\n")
            with open(scaled_valid_seq_pr_path, "a") as f:
                f.write(f"{scaled_valid_seq_pr}\n")
            with open(unscaled_valid_seq_pr_path, "a") as f:
                f.write(f"{unscaled_valid_seq_pr}\n")
            with open(log_ss_pf_path, "a") as f:
                f.write(f"{log_unscaled_ss_pf}\n")
            with open(times_path, "a") as f:
                f.write(f"{iter_time}\n")

            prev_nt_seq = curr_nt_seq
            prev_aa_seq = curr_aa_seq
            curr_nt_seq, curr_aa_seq, curr_pr_seq, curr_maxs = get_seqs(logits)

            curr_nt_seq_scaled = ''.join([curr_nt_seq[c_idx].lower() if curr_pr_seq[c_idx, curr_maxs[c_idx]] < 0.5 else curr_nt_seq[c_idx] for c_idx in range(len(curr_nt_seq))])
            diff_nt_str = ''.join(['-' if c1 == c2 else c2 for c1, c2 in zip(prev_nt_seq, curr_nt_seq)])
            diff_aa_str = ''.join(['-' if aa1 == aa2 else aa2 for aa1, aa2 in zip(prev_aa_seq, curr_aa_seq)])

            curr_coding_seq_oh = jnp.array(utils.seq_to_one_hot(curr_nt_seq[:dp.n_bases_coding]))
            curr_seq_cai = dp.unscaled_expected_cai_fn(curr_coding_seq_oh)

            with open(nuc_seq_path, "a") as f:
                f.write(f"{curr_nt_seq}\n")
            with open(argmax_cai_path, "a") as f:
                f.write(f"{curr_seq_cai}\n")
            with open(aa_seq_path, "a") as f:
                f.write(f"{curr_aa_seq}\n")
            with open(aa_seq_diff_path, "a") as f:
                f.write(f"{diff_aa_str}\n")
            with open(nuc_seq_diff_path, "a") as f:
                f.write(f"{diff_nt_str}\n")
            with open(aup_path, "a") as f:
                f.write(f"{dp.calc_aup(curr_nt_seq)}\n")
            with open(mfe_path, "a") as f:
                f.write(f"{dp.calc_mfe(curr_nt_seq)}\n")
            with open(efe_path, "a") as f:
                f.write(f"{dp.calc_efe(curr_nt_seq)}\n")

        if i % dp.args['save_pseq_every'] == 0:
            pseq_fpath = pseq_dir / f"pseq_i{i}.npy"
            jnp.save(pseq_fpath, p_seq, allow_pickle=False)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return



def get_argparse():
    parser = argparse.ArgumentParser(description="Optimize mRNA Sequence by maximizing EFE")

    # System setup
    parser.add_argument('-k', '--key-seed', type=int, default=0, help="Random key")
    parser.add_argument('--output-dir', type=str, default="output/",
                        help='Path to base output directory')
    parser.add_argument('--log-every', type=int, default=1,
                        help="Interval of iterations to log information")
    parser.add_argument('--save-pseq-every', type=int, default=50,
                        help="Frequency of saving pseqs.")
    parser.add_argument('--params-path', type=str, default=utils.TURNER_2004,
                        choices=[utils.TURNER_2004, utils.TURNER_1999, utils.ETERNA_185x],
                        help='Path to thermodynamic parameters')
    parser.add_argument('--fasta-path', type=str, default=utils.DATA_BASEDIR / "protein-fasta/P01588.fasta",
                        help='Path to FASTA file with target protein sequence')
    parser.add_argument('--run-name', type=str, nargs='?',
                        help='Name of run directory')
    parser.add_argument('--scale', type=float, nargs='?', help="Scale for Boltzmann rescaling")
    parser.add_argument('--loss-coeff', type=float, default=1.0,
                        help="Coefficient for loss")
    parser.add_argument('--checkpoint-every', type=int, default=7,
                        help="Checkpoint frequency")

    ## Optionally start from a given nuc. seq
    parser.add_argument('--start-nt-path', type=str, nargs='?',
                        help='Optional path to a FASTA file for initialization of logits')

    # Optimization parameters
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Num. iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.00001,
                        help="Learning rate for optimization")
    parser.add_argument('-o', '--optimizer', type=str, default="lamb",
                        help='Optimizer for gradient descent')
    parser.add_argument('--valid-seq-pr-thresh', type=float, default=0.5,
                        help="Threshold for valid sequence probability")

    ## CAI-specific
    parser.add_argument('--cai-threshold', type=float, default=0.8,
                        help="CAI value above which to scale. Must be in (0, 1)")
    parser.add_argument('--cai-scale', type=float, default=0.05,
                        help="Coefficient by which to scale CAI when above the provided threshold. Must be in [0, 1).")
    parser.add_argument('--cfs-table-path', type=str, default=protein.HOMOSAPIENS_CFS_PATH,
                        help='Path to codon frequency table')
    parser.add_argument('--no-cai', action='store_true',
                        help="If true, will ignore the CAI term in the geometric mean entirely")

    ## Neural-network-specific
    parser.add_argument('--nn-features', type=int, default=4000,
                        help="Number of features for the MLP")
    parser.add_argument('--nn-layers', type=int, default=6,
                        help="Number of layers for the MLP")
    parser.add_argument('--pretrain-lr', type=float, default=0.0001,
                        help="Learning rate for pretraining network")
    parser.add_argument('--pretrain-optimizer', type=str, default="adam",
                        help='Optimizer NN pretraining')
    parser.add_argument('--num-pretrain-iter', type=int, default=250,
                        help='Number of iterations for pretraining')



    return parser


if __name__ == "__main__":

    parser = get_argparse()
    args = vars(parser.parse_args())

    dp = MrnaDesignProblem(args)
    run(dp)
