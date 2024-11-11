""" This module contains code for dealing with amino acid sequence and coding sequence (CDS) data. """
import math
import dataclasses
import random
import pdb
import numpy as onp
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import tempfile
import re
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap

from jax_rnafold.common.utils import RNA_ALPHA
from jax_rnafold.common import utils




RES_ALPHA = "MGKTRADEYVLQWFSHNPCI"

CAI_DIR = utils.DATA_BASEDIR / "cai"
HOMOSAPIENS_CFS_PATH = CAI_DIR / "homosapiens.txt"

def aa_seq_to_one_hot(seq):
    all_vecs = list()
    for res in seq:
        res_idx = RES_ALPHA.index(res)
        res_vec = onp.zeros(20)
        res_vec[res_idx] = 1.0
        all_vecs.append(res_vec)
    return onp.array(all_vecs)

f64 = jnp.float64

AA_SINGLE_LETTER = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "End": "*",  # Stop codon
}
AA_ALPHA = list(AA_SINGLE_LETTER.values())
AA_ALPHA.remove("*")

def is_valid_aa_letter(aa):
    return aa in AA_SINGLE_LETTER.values() and aa != '*'

CODON_TABLE = {
    "F": ["UUC", "UUU"],
    "L": ["CUA", "CUC", "CUG", "CUU", "UUA", "UUG"],
    "I": ["AUA", "AUC", "AUU"],
    "M": ["AUG"],
    "V": ["GUA", "GUC", "GUG", "GUU"],
    "S": ["AGC", "AGU", "UCA", "UCC", "UCG", "UCU"],
    "P": ["CCA", "CCC", "CCG", "CCU"],
    "T": ["ACA", "ACC", "ACG", "ACU"],
    "A": ["GCA", "GCC", "GCG", "GCU"],
    "Y": ["UAC", "UAU"],
    "*": ["UAA", "UAG", "UGA"],
    "H": ["CAC", "CAU"],
    "Q": ["CAA", "CAG"],
    "N": ["AAC", "AAU"],
    "K": ["AAA", "AAG"],
    "D": ["GAC", "GAU"],
    "E": ["GAA", "GAG"],
    "C": ["UGC", "UGU"],
    "W": ["UGG"],
    "R": ["AGA", "AGG", "CGA", "CGC", "CGG", "CGU"],
    "G": ["GGA", "GGC", "GGG", "GGU"]
}
MAX_CODONS = 6

def get_ncodons(aa_seq):
    n_codons = list()
    for aa in aa_seq:
        n_codons.append(len(CODON_TABLE[aa]))
    return onp.array(n_codons)

def get_codon_matrix(aa_seq):
    codon_matrix = list()
    for aa in aa_seq:
        aa_codons = CODON_TABLE[aa]
        n_aa_codons = len(aa_codons)

        aa_codon_idxs = list()
        for codon in aa_codons:
            codon_idxs = [RNA_ALPHA.index(n) for n in codon]
            aa_codon_idxs.append(codon_idxs)

        for _ in range(MAX_CODONS - n_aa_codons):
            aa_codon_idxs.append([-1, -1, -1])

        codon_matrix.append(aa_codon_idxs)

    codon_matrix = onp.array(codon_matrix)
    return codon_matrix


def get_random_codon_pseq(aa_seq):
    n_aa = len(aa_seq)
    pseq_codon = onp.empty((n_aa, MAX_CODONS), dtype=onp.float64)

    for i in range(n_aa):
        n_codons = len(CODON_TABLE[aa_seq[i]])
        codon_probs = onp.random.random_sample(n_codons)
        codon_probs /= onp.sum(codon_probs)

        codon_probs = onp.pad(codon_probs, (0, MAX_CODONS - n_codons))

        pseq_codon[i] = codon_probs

    return pseq_codon

def get_random_codon_pseq_onehot(aa_seq):
    n_aa = len(aa_seq)
    pseq_codon = onp.empty((n_aa, MAX_CODONS), dtype=onp.float64)

    for i in range(n_aa):
        n_codons = len(CODON_TABLE[aa_seq[i]])
        codon_probs = onp.zeros(MAX_CODONS)
        codon_choice = random.choice(range(n_codons))
        codon_probs[codon_choice] = 1.0

        pseq_codon[i] = codon_probs

    return pseq_codon


class CodonFrequencyTable:
    """
    Loads a codon frequency table from a file and provides methods for computing metrics with respect to the loaded frequencies.

    Attributes:
      table_path: The file path to the codon frequency table.
    """
    def __init__(self, table_path: str):
        file = open(table_path, 'r')
        self.codons = set()
        self.codon_to_aa = {}
        self.aa_to_codons = {}
        self.codon_freq = {}
        self.aa_max_freq = {}
        # The format uses T instead of U
        self.nt_map = {'A': 'A', 'T': 'U', 'C': 'C', 'G': 'G'}
        for line in file:
            tokens = line.strip(" \n").split()
            if len(tokens) != 3:
                continue
            aa = tokens[0]
            aa = AA_SINGLE_LETTER[aa]
            codon = ''.join([self.nt_map[nt] for nt in tokens[1]])
            freq = round(float(tokens[2]))
            self.codons.add(codon)
            self.codon_to_aa[codon] = aa
            if aa not in self.aa_to_codons:
                self.aa_to_codons[aa] = set()
            self.aa_to_codons[aa].add(codon)
            self.codon_freq[codon] = freq
            if aa not in self.aa_max_freq:
                self.aa_max_freq[aa] = 0
            self.aa_max_freq[aa] = max(self.aa_max_freq[aa], freq)
        file.close()

    def get_codon_freq(self, codon: str) -> float:
        return self.codon_freq[codon]

    def get_aa_max_freq(self, aa):
        return self.aa_max_freq[aa]

    def get_codons(self, aa) -> set[str]:
        return self.aa_to_codons[aa]

    # Maximum number of codons for a single amino acid
    def max_codons(self) -> int:
        return max(len(self.get_codons(aa)) for aa in self.aa_to_codons)

    def get_aa(self, codon):
        return self.codon_to_aa[codon]

    def codon_adaption_weight(self, codon):
        return self.get_codon_freq(codon) / self.get_aa_max_freq(self.get_aa(codon))

    def codon_adaptation_index(self, cds: list) -> float:
        """
        Computes the codon adaptation sequence (CAI) of a given coding sequence.

        Args:
          cds: A coding sequence represented as a list of triplets.

        Returns:
          The CAI of the given coding sequence.
        """
        cai = 1
        for codon in cds:
            cai *= self.codon_adaption_weight(codon)
        return cai**(1/len(cds))

    def log_codon_adaptation_index(self, cds):
        cai = 0
        for codon in cds:
            cai += math.log(self.codon_adaption_weight(codon))
        return cai / len(cds)

    def nuc_seq_to_aa_seq(self, nuc_seq):
        n_bases = len(nuc_seq)
        assert(n_bases % 3 == 0)
        n_aa = n_bases // 3
        assert(all([nuc in RNA_ALPHA for nuc in nuc_seq]))

        aa_seq = ""
        for i in range(n_aa):
            codon = nuc_seq[i*3:i*3+3]
            aa = self.get_aa(codon)
            aa_seq += aa
        return aa_seq

    def get_uniform_codon_logits(self, aa_seq: str) -> onp.ndarray:
        """
        Given an amino acid sequence, returns a probabilistic nucleotide
        sequence that represents the uniform distribution of valid codons.

        Args:
          aa_seq: The amino acid sequence of length `n`.

        Returns:
          A probabilistic sequence of shape `[n, 4]`.
        """
        pseq = onp.zeros((len(aa_seq)*3, 4))
        idx = 0
        for aa in aa_seq:
            aa_codons = self.get_codons(aa)
            for i in range(3):
                for codon in aa_codons:
                    pseq[idx+i, RNA_ALPHA.index(codon[i])] += 1
            idx += 3
        return pseq





@dataclasses.dataclass
class UniProtAASeq:
    seq: str
    uniprot_name: str
    protein_name: str


def read_cds(tsv_path):
    cds = []
    first = True
    for line in open(tsv_path, 'r'):
        if first:
            first = False
            continue
        tokens = line.strip(" \n").split("\t")
        if len(tokens) != 7:
            assert False, "Invalid CDS file format"
        if any(not is_valid_aa_letter(aa) for aa in tokens[6]):
            continue
        cds.append(UniProtAASeq(tokens[6], tokens[1], tokens[2]))
    return cds


def get_rand_aa_seq(n: int, key=None) -> str:
    """
    Samples a random discrete protein sequence of length `n`.

    Args:
      n: A positive integer defining the length of the desired protein sequence.

    Returns:
      A random discrete protein sequence of length `n`.
    """
    if key is not None:
        random.seed(key)
    rand_aa_seq = [random.choice(AA_ALPHA) for _ in range(n)]
    rand_aa_seq = "".join([str(aa) for aa in rand_aa_seq])
    return rand_aa_seq

def random_cds(aa_seq: str, freq_table: CodonFrequencyTable) -> list:
    """
    Sample a random coding sequence for a given protein.

    Args:
      aa_seq: The amino acid sequence.
      freq_table: A coding frequency table.

    Returns:
      A list of `n` triplets where the :math:`i^{th}` triplet codes for the :math:`i^{th}` residue.
    """
    cds = []
    for aa in aa_seq:
        cds.append(random.choice(list(freq_table.aa_to_codons[aa])))
    return cds



def fast_expected_cai(codon_choices):
    ans = 1
    for i in range(len(codon_choices)):
        ev = 0
        for w, pr in codon_choices[i]:
            ev += w**(1/len(codon_choices)) * pr
        ans *= ev
    return ans

def brute_force_expected_cai(codon_choices):
    sm = 0
    def f(i, choices):
        nonlocal sm
        if i == len(codon_choices):
            cai = 1
            prob = 1
            for ii in range(len(choices)):
                w, pr = codon_choices[ii][choices[ii]]
                cai *= w
                prob *= pr
            cai **= 1/len(codon_choices)
            sm += cai*prob
            return
        for j in range(len(codon_choices[i])):
            f(i+1, choices + [j])
    f(0, [])
    return sm


def pseq_to_codon_choices(pseq, cfs):
    n_nt = pseq.shape[0]
    assert(n_nt % 3 == 0)
    n_aa = n_nt // 3

    codon_choices = list()
    for res_idx in range(n_aa):
        pnuc1 = pseq[res_idx*3]
        pnuc2 = pseq[res_idx*3 + 1]
        pnuc3 = pseq[res_idx*3 + 2]

        ith_codon_choices = list()

        for nt1_idx in utils.N4:
            nt1 = RNA_ALPHA[nt1_idx]
            for nt2_idx in utils.N4:
                nt2 = RNA_ALPHA[nt2_idx]
                for nt3_idx in utils.N4:
                    nt3 = RNA_ALPHA[nt3_idx]
                    codon = nt1 + nt2 + nt3

                    pr_codon = pnuc1[nt1_idx]*pnuc2[nt2_idx]*pnuc3[nt3_idx]
                    codon_weight = cfs.codon_adaption_weight(codon)
                    ith_codon_choices.append((codon_weight, pr_codon))

        codon_choices.append(ith_codon_choices)

    return codon_choices


def pseq_to_target_codon_choices(pseq, target, cfs):
    assert(is_valid_aa_letter(aa) for aa in target)

    n_aa = len(target)
    n_bases = pseq.shape[0]
    assert(n_bases / 3 == n_aa)

    codon_choices = list()
    for aa_idx, aa in enumerate(target):
        nuc1 = pseq[aa_idx*3]
        nuc2 = pseq[aa_idx*3 + 1]
        nuc3 = pseq[aa_idx*3 + 2]

        aa_codons = cfs.get_codons(aa)
        ith_codon_choices = list()
        for codon in aa_codons:
            # Get codon probability
            codon_pr = nuc1[RNA_ALPHA.index(codon[0])] * nuc2[RNA_ALPHA.index(codon[1])] * nuc3[RNA_ALPHA.index(codon[2])]

            # Get codon weight
            codon_weight = cfs.codon_adaption_weight(codon)

            ith_codon_choices.append((codon_weight, codon_pr))
        codon_choices.append(ith_codon_choices)

    return codon_choices


def expected_cai_for_target(prob_seq, target, cfs):
    codon_choices = pseq_to_target_codon_choices(prob_seq, target, cfs)
    return fast_expected_cai(codon_choices)


def valid_seq_prob(prob_seq, target, cfs):
    codon_choices = pseq_to_target_codon_choices(prob_seq, target, cfs)
    valid_seq_pr = 1.0
    n_codons = len(codon_choices)
    codon_probs = list()
    for i in range(n_codons):
        valid_codon_pr = 0
        for w, pr in codon_choices[i]:
            valid_codon_pr += pr
        valid_seq_pr *= valid_codon_pr
        codon_probs.append(valid_codon_pr)
    return valid_seq_pr, codon_probs


all_triplets = list() # Same order as two tensor products
for n1 in RNA_ALPHA:
    for n2 in RNA_ALPHA:
        for n3 in RNA_ALPHA:
            all_triplets.append(n1+n2+n3)

def get_ecai_weights_table(aa_seq: str, cfs: CodonFrequencyTable, restrict: bool = True, invalid_weight: float = 0.0):
    n_aa = len(aa_seq)

    weights_table = list() # will be n_aa x 64
    for i in range(n_aa):
        ith_row = list()
        aa = aa_seq[i]
        aa_codons = cfs.get_codons(aa)
        for codon in all_triplets:
            if (codon in aa_codons) or not restrict:
                ith_row.append(cfs.codon_adaption_weight(codon))
            else:
                ith_row.append(invalid_weight)
        weights_table.append(ith_row)
    weights_table = jnp.array(weights_table, dtype=f64)

    return weights_table


def get_expected_cai_fn(aa_seq: str, cfs: CodonFrequencyTable,
                        invalid_weight: float = 0.0, restrict: bool = True) -> Callable:
    """
    Returns a JAX-compatible function for computing the expected CAI
    of a probabilistic sequence.

    Args:
      aa_seq: The target amino acid sequence.
      cfs: A codon frequency table.
      invalid_weight: The weight assigned to invalid codons.
      restrict: If true, will only compute eCAI per the codons of the target (ignoring off-target codons). If False, `aa_seq` and `invalid-weight` are ignored.

    Returns:
      A function for computing the expected CAI of a probabilistic sequence
      for the target amino acid sequence.
    """

    assert(is_valid_aa_letter(aa) for aa in aa_seq)
    n_aa = len(aa_seq)

    weights_table = get_ecai_weights_table(aa_seq, cfs, restrict, invalid_weight) # will be n_aa x 64

    def compute_cai_term(w, pr):
        return w**(1/n_aa) * pr

    def compute_codon_ev(prob_seq, i):
        nuc1 = prob_seq[i*3]
        nuc2 = prob_seq[i*3 + 1]
        nuc3 = prob_seq[i*3 + 2]

        triplet_probs = jnp.kron(jnp.kron(nuc1, nuc2), nuc3)
        weights = weights_table[i]

        codon_ev = jnp.sum(vmap(compute_cai_term, (0, 0))(weights, triplet_probs))
        return codon_ev

    def expected_cai(prob_seq):
        codon_evs = vmap(compute_codon_ev, (None, 0))(prob_seq, jnp.arange(n_aa))
        return jnp.prod(codon_evs)
    return expected_cai


def get_expected_cai_fn_general(n_aa):

    def compute_cai_term(w, pr):
        return w**(1/n_aa) * pr

    def compute_codon_ev(prob_seq, weights_table, i):
        nuc1 = prob_seq[i*3]
        nuc2 = prob_seq[i*3 + 1]
        nuc3 = prob_seq[i*3 + 2]

        triplet_probs = jnp.kron(jnp.kron(nuc1, nuc2), nuc3)
        weights = weights_table[i]

        codon_ev = jnp.sum(vmap(compute_cai_term, (0, 0))(weights, triplet_probs))
        return codon_ev

    def expected_cai(prob_seq, weights_table):
        codon_evs = vmap(compute_codon_ev, (None, None, 0))(prob_seq, weights_table, jnp.arange(n_aa))
        return jnp.prod(codon_evs)
    return expected_cai


def get_valid_seq_table(aa_seq, cfs):
    n_aa = len(aa_seq)

    valid_table = list() # will be n_aa x 64
    for i in range(n_aa):
        ith_row = list()
        aa = aa_seq[i]
        aa_codons = cfs.get_codons(aa)
        for codon in all_triplets:
            ith_row.append(int(codon in aa_codons))
        valid_table.append(ith_row)
    valid_table = jnp.array(valid_table, dtype=jnp.int32)

    return valid_table


def get_valid_seq_pr_fn(aa_seq: str, cfs: CodonFrequencyTable):

    assert(is_valid_aa_letter(aa) for aa in aa_seq)
    n_aa = len(aa_seq)

    valid_table = get_valid_seq_table(aa_seq, cfs) # will be n_aa x 64

    def compute_valid_codon_prob(prob_seq, i):
        nuc1 = prob_seq[i*3]
        nuc2 = prob_seq[i*3 + 1]
        nuc3 = prob_seq[i*3 + 2]
        triplet_probs = jnp.kron(jnp.kron(nuc1, nuc2), nuc3)
        valid_codon_probs = jnp.multiply(triplet_probs, valid_table[i])
        return valid_codon_probs.sum()

    def valid_seq_prob(prob_seq):
        valid_codon_probs = vmap(compute_valid_codon_prob, (None, 0))(prob_seq, jnp.arange(n_aa))
        return jnp.prod(valid_codon_probs)
    return valid_seq_prob


def get_valid_seq_pr_fn_general(n_aa):

    def compute_valid_codon_prob(prob_seq, valid_table, i):
        nuc1 = prob_seq[i*3]
        nuc2 = prob_seq[i*3 + 1]
        nuc3 = prob_seq[i*3 + 2]
        triplet_probs = jnp.kron(jnp.kron(nuc1, nuc2), nuc3)
        valid_codon_probs = jnp.multiply(triplet_probs, valid_table[i])
        return valid_codon_probs.sum()

    def valid_seq_prob(prob_seq, valid_table):
        valid_codon_probs = vmap(compute_valid_codon_prob, (None, None, 0))(prob_seq, valid_table, jnp.arange(n_aa))
        return jnp.prod(valid_codon_probs)
    return valid_seq_prob



DEFAULT_CONFIG_TEMPLATE = utils.DATA_BASEDIR / "misc" / "template.config"
# example: run_codon_folding("MEKSFVITDPWLPDYPIISASDGFLELTE", "../max-ld/build/exe/fold_codon_graph")
def run_codon_folding(sequence: str, executable_path: str, config_template: str = DEFAULT_CONFIG_TEMPLATE) -> dict:

    # Create a temporary directory to hold the config file
    with tempfile.TemporaryDirectory() as temp_dir:

        # Path to the new config file
        config_file_path = os.path.join(temp_dir, 'temp_config.config')

        # Read the template configuration file
        with open(config_template, 'r') as template_file:
            config_content = template_file.read()

        # Replace the placeholder with the new sequence
        config_content = config_content.replace('SEQUENCE', sequence)

        # Write the modified config file to the temporary directory
        with open(config_file_path, 'w') as temp_config_file:
            temp_config_file.write(config_content)

        # Call the executable and capture the output
        result = subprocess.run([executable_path, config_file_path], capture_output=True, text=True)

        # Check if the process ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"Error running the executable: {result.stderr}")

        # Extract the last four lines from the output
        output_lines = result.stdout.splitlines()
        sequence = output_lines[-4].strip()
        structure = output_lines[-3].strip()
        efe_line = output_lines[-2].strip()
        cai_line = output_lines[-1].strip()

        # Extract EFE and CAI using regex
        efe_match = re.search(r"EFE: (-?\d+\.\d+)", efe_line)
        cai_match = re.search(r"CAI: (\d+\.\d+)", cai_line)

        if efe_match and cai_match:
            efe = float(efe_match.group(1))
            cai = float(cai_match.group(1))
        else:
            raise ValueError("Could not extract EFE and CAI from output")

        # Return the parsed values in a dictionary
        return {
            'sequence': sequence,
            'structure': structure,
            'EFE': efe,
            'CAI': cai
        }




if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    table_path = HOMOSAPIENS_CFS_PATH
    cfs = CodonFrequencyTable(table_path)

    """
    # Some testing for probability of sampling a valid sequence


    ## Compare our JAX valid prob calculation to our pythonic one
    n_aa = 50
    n_nt = n_aa*3
    aa_seq = get_rand_aa_seq(n_aa)
    valid_seq_pr_fn = jax.jit(get_valid_seq_pr_fn(aa_seq, cfs))

    pseq_random = jnp.array(utils.random_pseq(n_nt)) # A random pseq
    random_prob, _ = valid_seq_prob(pseq_random, aa_seq, cfs)
    jax_random_prob = valid_seq_pr_fn(pseq_random)

    cds = ''.join(random_cds(aa_seq, cfs)) # A one-hot CDS
    pseq_cds = jnp.array(utils.seq_to_one_hot(cds))
    cds_prob, _ = valid_seq_prob(pseq_cds, aa_seq, cfs)
    jax_cds_prob = valid_seq_pr_fn(pseq_cds)

    uniform_logits = jnp.array(cfs.get_uniform_codon_logits(aa_seq)) * 10

    pseq_uniform_softmax = jax.nn.softmax(uniform_logits)
    uniform_softmax_prob, _ = valid_seq_prob(pseq_uniform_softmax, aa_seq, cfs)
    jax_uniform_softmax_prob = valid_seq_pr_fn(pseq_uniform_softmax)

    pseq_uniform_norm = uniform_logits / uniform_logits.sum(axis=1, keepdims=True)
    uniform_norm_prob, _ = valid_seq_prob(pseq_uniform_norm, aa_seq, cfs)
    jax_uniform_norm_prob = valid_seq_pr_fn(pseq_uniform_norm)


    pdb.set_trace()



    ## Construct some function based on a threshold

    threshold = 0.5
    poly_coeff = 50
    linear_coeff = 0.025
    def scale_fn(prob):
        return jnp.where(prob >= threshold, threshold + linear_coeff*(prob - threshold), -poly_coeff*(prob-threshold)**2 + threshold)

    # test_vals = onp.linspace(0, 1, 100)
    test_vals = onp.linspace(0.45, 0.55, 1000)
    plt.plot(test_vals, scale_fn(test_vals))
    plt.xlabel("Raw probability")
    plt.ylabel("Scaled probability")
    plt.show()

    pdb.set_trace() # TODO
    """



    ## Check our valid probability calculation via random sampling
    n_aa = 50
    n_nt = n_aa*3
    aa_seq = get_rand_aa_seq(n_aa)

    nsamples = 1000
    def sample_valid_prob(pseq, key):
        sampled_seqs, _ = utils.sample_discrete_seqs(pseq, nsamples, key)
        valid = list()
        for seq in sampled_seqs:
            sampled_aa_seq = cfs.nuc_seq_to_aa_seq(seq)
            valid.append(int(sampled_aa_seq == aa_seq))

        running_avg = onp.cumsum(valid) / onp.arange(1, nsamples+1)
        return onp.mean(valid), running_avg

    pseq_random = jnp.array(utils.random_pseq(n_nt)) # A random pseq
    random_prob, _ = valid_seq_prob(pseq_random, aa_seq, cfs)
    key, random_key = jax.random.split(key)
    sampled_prob_random, running_sampled_prob_random = sample_valid_prob(pseq_random, random_key)

    cds = ''.join(random_cds(aa_seq, cfs)) # A one-hot CDS
    pseq_cds = jnp.array(utils.seq_to_one_hot(cds))
    cds_prob, _ = valid_seq_prob(pseq_cds, aa_seq, cfs)
    key, cds_key = jax.random.split(key)
    sampled_prob_cds, running_sampled_prob_cds = sample_valid_prob(pseq_cds, random_key)

    uniform_logits = jnp.array(cfs.get_uniform_codon_logits(aa_seq)) * 10

    pseq_uniform_softmax = jax.nn.softmax(uniform_logits)
    uniform_softmax_prob, _ = valid_seq_prob(pseq_uniform_softmax, aa_seq, cfs)
    key, softmax_key = jax.random.split(key)
    sampled_prob_softmax, running_sampled_prob_softmax = sample_valid_prob(pseq_uniform_softmax, random_key)

    pseq_uniform_norm = uniform_logits / uniform_logits.sum(axis=1, keepdims=True)
    uniform_norm_prob, _ = valid_seq_prob(pseq_uniform_norm, aa_seq, cfs)
    key, norm_key = jax.random.split(key)
    sampled_prob_norm, running_sampled_prob_norm = sample_valid_prob(pseq_uniform_norm, random_key)


    plt.plot(running_sampled_prob_softmax, label="Sampled softmax prob", color="blue")
    plt.axhline(y=uniform_softmax_prob, linestyle="--", label="Computed softmax prob", color="blue")
    plt.plot(running_sampled_prob_norm, label="Sampled norm prob", color="green")
    plt.axhline(y=uniform_norm_prob, linestyle="--", label="Computed norm prob", color="green")
    plt.plot(running_sampled_prob_random, label="Sampled random prob", color="red")
    plt.axhline(y=random_prob, linestyle="--", label="Computed random prob", color="red")


    plt.title(f"Convergence of valid seq probability via sampling, n_aa={n_aa}")
    plt.ylabel("Probability")
    plt.xlabel("Num nuc. seq. samples")

    plt.legend()
    plt.tight_layout()
    plt.show()


    pdb.set_trace()




    ## Get plots of probabilities as a function of protein length for different methods of pseq initialization
    mean_norm_probs = list()
    mean_softmax_probs = list()
    mean_random_probs = list()
    mean_cds_probs = list()
    nsamples = 10
    n_aas = [1, 3, 5, 10, 15, 25, 50, 100]
    for n_aa in tqdm(n_aas):
        n_nt = n_aa*3

        uniform_norm_probs = list()
        uniform_softmax_probs = list()
        cds_probs = list()
        random_probs = list()

        for _ in range(nsamples):
            aa_seq = get_rand_aa_seq(n_aa)
            pseq = jnp.array(utils.random_pseq(n_nt)) # A random pseq
            cds = ''.join(random_cds(aa_seq, cfs)) # A one-hot CDS
            pseq_cds = jnp.array(utils.seq_to_one_hot(cds))

            uniform_logits = jnp.array(cfs.get_uniform_codon_logits(aa_seq)) * 10
            pseq_uniform_softmax = jax.nn.softmax(uniform_logits)
            pseq_uniform_norm = uniform_logits / uniform_logits.sum(axis=1, keepdims=True)

            uniform_norm_prob, uniform_norm_codon_probs = valid_seq_prob(pseq_uniform_norm, aa_seq, cfs)
            uniform_norm_probs.append(uniform_norm_prob)
            print(f"Uniform logits, norm: {uniform_norm_prob}")

            uniform_softmax_prob, uniform_softmax_codon_probs = valid_seq_prob(pseq_uniform_softmax, aa_seq, cfs)
            uniform_softmax_probs.append(uniform_softmax_prob)
            print(f"Uniform logits, softmax: {uniform_softmax_prob}")

            cds_prob, cds_codon_probs = valid_seq_prob(pseq_cds, aa_seq, cfs)
            cds_probs.append(cds_prob)
            print(f"CDS (one-hot): {cds_prob}")

            random_prob, random_codon_probs = valid_seq_prob(pseq, aa_seq, cfs)
            random_probs.append(random_prob)
            print(f"Random: {random_prob}")

        mean_norm_probs.append(onp.mean(uniform_norm_probs))
        mean_softmax_probs.append(onp.mean(uniform_softmax_probs))
        mean_random_probs.append(onp.mean(random_probs))
        mean_cds_probs.append(onp.mean(cds_probs))

    plt.plot(n_aas, onp.log(mean_norm_probs), label="Normalize")
    plt.plot(n_aas, onp.log(mean_softmax_probs), label="Softmax")
    # plt.plot(n_aas, onp.log(mean_random_probs), label="Random")
    plt.xlabel("# Residues")
    plt.ylabel("Avg. Log. Prob. of Sampling Valid Seq")
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(n_aas, onp.log(mean_norm_probs), label="Normalize")
    plt.plot(n_aas, onp.log(mean_softmax_probs), label="Softmax")
    plt.plot(n_aas, onp.log(mean_random_probs), label="Random")
    plt.xlabel("# Residues")
    plt.ylabel("Avg. Log. Prob. of Sampling Valid Seq")
    plt.legend()
    plt.show()


    pdb.set_trace()



    # Some testing for expected CAI
    key = jax.random.PRNGKey(0)

    """
    n_aa = 2
    n_nt = n_aa*3
    pseq = jnp.array(utils.random_pseq(n_nt))
    aa_seq = get_rand_aa_seq(n_aa)
    """

    pseq = jnp.load("/home/ryan/Downloads/for-max/cai0.8/mev/pseq_nn.npy")
    n_nt = pseq.shape[0]
    assert(n_nt % 3 == 0)
    n_aa = n_nt // 3
    aa_seq = "MGGSGGSGYQPYRVVVLGGSGGSPYRVVVLSFGGSGGSLSPRWYFYY"
    assert(len(aa_seq) == n_aa)


    nsamples = 5000
    seqs, _ = utils.sample_discrete_seqs(pseq, nsamples, key)


    table_path = HOMOSAPIENS_CFS_PATH
    cfs = CodonFrequencyTable(table_path)

    codon_choices = pseq_to_codon_choices(pseq, cfs)
    target_codon_choices = pseq_to_target_codon_choices(pseq, aa_seq, cfs)
    pdb.set_trace()


    # Three ways of computing CAI

    ## Method 1: Random sampling sequences
    all_sampled_cais = list()
    running_avg = list()
    for seq in tqdm(seqs, desc="Computing sampled CAIs"):
        cds = [seq[res_idx*3:res_idx*3+3] for res_idx in range(n_aa)]
        sampled_cai = cfs.codon_adaptation_index(cds)
        all_sampled_cais.append(sampled_cai)
    mean_sampled_cai = onp.mean(all_sampled_cais)
    cai_running_avg = onp.cumsum(all_sampled_cais) / onp.arange(1, nsamples+1)



    full_ecai = fast_expected_cai(codon_choices)
    restricted_ecai = fast_expected_cai(target_codon_choices)

    plt.plot(cai_running_avg, label="Sampled running average")
    plt.axhline(y=full_ecai, linestyle="--", color="green", label="Full")
    plt.axhline(y=restricted_ecai, linestyle="--", color="red", label="Restricted")
    plt.ylabel("Expected CAI")
    plt.xlabel("Number of sampled sequences")
    plt.title("Comparison using non-JAX eCAI function")
    plt.legend()
    plt.show()
    plt.clf()

    invalid_cai_weight = 1e-3
    ecai_fn_full = get_expected_cai_fn(aa_seq, cfs, invalid_weight=invalid_cai_weight, restrict=False)
    ecai_fn_restricted = get_expected_cai_fn(aa_seq, cfs, invalid_weight=invalid_cai_weight, restrict=True)

    full_ecai_jax = ecai_fn_full(pseq)
    restricted_ecai_jax = ecai_fn_restricted(pseq)


    plt.plot(cai_running_avg, label="Sampled running average")
    plt.axhline(y=full_ecai_jax, linestyle="--", color="green", label="Full")
    plt.axhline(y=restricted_ecai_jax, linestyle="--", color="red", label="Restricted")
    plt.ylabel("Expected CAI")
    plt.xlabel("Number of sampled sequences")
    plt.title("Comparison using JAX eCAI function")
    plt.legend()
    plt.show()
    plt.clf()


    ## Sampling valid sampling probability
    nsamples = 2500
    sampled_seqs, _ = utils.sample_discrete_seqs(pseq, nsamples, key)
    valid = list()
    for seq in tqdm(sampled_seqs):
        sampled_aa_seq = cfs.nuc_seq_to_aa_seq(seq)
        valid.append(int(sampled_aa_seq == aa_seq))

    running_avg = onp.cumsum(valid) / onp.arange(1, nsamples+1)

    valid_seq_pr_fn = jax.jit(get_valid_seq_pr_fn(aa_seq, cfs))
    valid_pr = valid_seq_pr_fn(pseq)
    plt.plot(onp.arange(1, nsamples+1), running_avg, label="Sampled running avg.")
    plt.axhline(y=valid_pr, linestyle="--", label="calculated", color="green")
    plt.xlabel("# sampled sequences")
    plt.ylabel("Prob. sampling valid sequence")
    plt.legend()
    plt.show()


    pdb.set_trace()
    print("done")
