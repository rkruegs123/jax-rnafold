import unittest
import pdb
import numpy as onp
from abc import ABC, abstractmethod
from typing import Dict

import jax.numpy as jnp
from jax_md import dataclasses, util

from jax_rnafold.common.utils import RNA_ALPHA, INVALID_BASE, structure_tree, ALL_PAIRS, NON_GC_PAIRS, DEFAULT_HAIRPIN
from jax_rnafold.common.utils import non_gc_pairs_mat, all_pairs_mat
from jax_rnafold.common.utils import kb, CELL_TEMP, MAX_PRECOMPUTE, TURNER_2004
from jax_rnafold.common.read_vienna_params import NNParams
from jax_rnafold.common.energy_hash import float_hash, get_jax_float_hash_fn
from jax_rnafold.common.utils import boltz_onp, boltz_jnp


Array = util.Array

DEFAULT_SPECIAL_HAIRPINS = ["AAA", "AAAA", "AAAAAA"]
allowed_sp_hairpin_lens = set([3, 4, 6])
for sp_hairpin in DEFAULT_SPECIAL_HAIRPINS:
    len_hairpin = len(sp_hairpin)
    assert(len_hairpin in allowed_sp_hairpin_lens)


class Model(ABC):
    """
    An abstract base class for representing an energy model. An energy model
    contains methods for computing the energy contributions of various
    nearest neighbor motifs, and stores metadata such as special hairpins
    considered in the given energy model.

    Attributes:
      temp (float): The temperature in Kelvin.
      beta (float): The inverse temperature.
      hairpin (int): The minimum hairpin size.
      jaxify (bool): If True, will ensure that various methods and metadata are JAX-compatible.
      special_hairpins (list[str]): A list of special hairpins.
      special_hairpin_energies (list[float]): The energies corresponding to each special hairpin.
      n_special_hairpins (int): The number of special hairpins.
      special_hairpin_lens (list[int]): The length of each special hairpin.
      special_hairpin_idxs (list[int]): The concatenated base identities of each special hairpin.
      special_hairpin_start_pos (list[int]): The start positions of each special hairpin in `special_hairpin_idxs`.
    """

    temp: float # Kelvin
    beta: float
    hairpin: int

    jaxify: bool

    special_hairpins: list[str]
    special_hairpin_energies: list[float]

    n_special_hairpins: int
    special_hairpin_lens: list[int] # array 1: length of each special hairpin
    special_hairpin_idxs: list[int] # array 2: all characters concatenated
    special_hairpin_start_pos: list[int] # array 3: start position for each in array 2


    def process_special_hairpins(self):
        self.n_special_hairpins = len(self.special_hairpins)
        special_hairpin_lens = [len(sp_hairpin) for sp_hairpin in self.special_hairpins]

        special_hairpin_idxs = list()
        special_hairpin_start_pos = list()
        idx = 0
        for sp_hairpin in self.special_hairpins:
            special_hairpin_start_pos.append(idx)
            for nuc in sp_hairpin:
                special_hairpin_idxs.append(RNA_ALPHA.index(nuc))
                idx += 1

        if self.jaxify:
            self.special_hairpin_lens = jnp.array(special_hairpin_lens)
            self.special_hairpin_idxs = jnp.array(special_hairpin_idxs)
            self.special_hairpin_start_pos = jnp.array(special_hairpin_start_pos)
            self.special_hairpin_energies = jnp.array(self.special_hairpin_energies)
        else:
            self.special_hairpin_lens = onp.array(special_hairpin_lens)
            self.special_hairpin_idxs = onp.array(special_hairpin_idxs)
            self.special_hairpin_start_pos = onp.array(special_hairpin_start_pos)

        return

    @abstractmethod
    def en_ext_branch(self, bi: int, bj: int) -> float:
        """Computes the energy contribution of an exterior loop branch."""
        pass

    @abstractmethod
    def en_multi_branch(self, bi: int, bk: int) -> float:
        """Computes the energy contribution of an multiloop branch."""
        pass

    @abstractmethod
    def en_multi_closing(self, bi: int, bj: int) -> float:
        """Computes the closing penalty of an multiloop branch."""
        pass

    @abstractmethod
    def en_multi_unpaired(self) -> float:
        """Computes the energy of an unpaired nucleotide in a multiloop branch."""
        pass

    @abstractmethod
    def en_hairpin_not_special(self, bi: int, bj: int, bip1: int, bjm1: int, nunpaired: int) -> float:
        """Computes the energy of a generic hairpin."""
        # Note that vienna ignores the all-C case
        pass

    @abstractmethod
    def en_hairpin_special(self, id: int) -> float:
        """Computes the energy of a special hairpin."""
        # id is the index into SPECIAL_HAIRPINS
        pass

    @abstractmethod
    def en_stack(self, bi: int, bj: int, bk: int, bl: int) -> float:
        """Computes the energy of a pair of stacked base pairs."""
        pass

    @abstractmethod
    def en_bulge(self, bi: int, bj: int, bk: int, bl: int, nunpaired: int) -> float:
        """Computes the energy of a bulge loop."""
        pass

    @abstractmethod
    def en_il_inner_mismatch(self, bi: int, bj: int, bip1: int, bjm1: int) -> float:
        """Computes the energy of the inner mismatch of an internal loop."""
        pass

    @abstractmethod
    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1) -> float:
        """Computes the energy of the outer mismatch of an internal loop."""
        pass

    @abstractmethod
    def en_internal_init(self, sz) -> float:
        """Computes the initiation cost of an internal loop."""
        pass

    @abstractmethod
    def en_internal_asym(self, lup, rup) -> float:
        """Computes the asymmetry cost of an internal loop."""
        pass

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup) -> float:
        """Computes the energy of a generic internal loop."""
        en = self.en_internal_init(lup+rup)*self.en_internal_asym(lup, rup)*self.en_il_inner_mismatch(
            bi, bj, bip1, bjm1)*self.en_il_outer_mismatch(bk, bl, bkm1, blp1)
        return en


class All0Model(Model):
    def __init__(self, temp=CELL_TEMP, jaxify=False, hairpin=DEFAULT_HAIRPIN):
        self.special_hairpins = DEFAULT_SPECIAL_HAIRPINS
        self.special_hairpin_energies = [0.0 for _ in self.special_hairpins]

        self.jaxify = jaxify
        self.process_special_hairpins()

        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin


    def en_ext_branch(self, bi, bj):
        return 0

    def en_multi_branch(self, bi, bk):
        return 0

    def en_multi_closing(self, bi, bj):
        return 0

    def en_multi_unpaired(self):
        return 0

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 0

    def en_hairpin_special(self, id):
        return 0

    def en_stack(self, bi, bj, bk, bl):
        return 0

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return 0

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 0

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 0

    def en_internal_init(self, sz):
        return 0

    def en_internal_asym(self, lup, rup):
        return 0


class All1Model(Model):
    """
    An energy model that assigns an energy of 1 to each motif.

    Attributes:
      temp: The temperature in Kelvin
      hairpin: The minimum hairpin size.
      jaxify: If True, will ensure that various methods and metadata are JAX-compatible.
    """
    def __init__(self, temp=CELL_TEMP, jaxify=False, hairpin=DEFAULT_HAIRPIN):
        self.special_hairpins = DEFAULT_SPECIAL_HAIRPINS
        self.special_hairpin_energies = [1.0 for _ in self.special_hairpins]

        self.jaxify = jaxify
        self.process_special_hairpins()

        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin

    def en_ext_branch(self, bi, bj):
        return 1

    def en_multi_branch(self, bi, bk):
        return 1

    def en_multi_closing(self, bi, bj):
        return 1

    def en_multi_unpaired(self):
        return 1

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 1

    def en_hairpin_special(self, id):
        return 1

    def en_stack(self, bi, bj, bk, bl):
        return 1

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return 1

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 1

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 1

    def en_internal_init(self, sz):
        return 1

    def en_internal_asym(self, lup, rup):
        return 1



class RandomModel(Model):
    """
    An energy model that assigns an energy of 1 to each motif.

    Attributes:
      temp: The temperature in Kelvin
      hairpin: The minimum hairpin size.
      jaxify: If True, will ensure that various methods and metadata are JAX-compatible.
      seed: A random seed.
    """
    def __init__(self, temp: float = CELL_TEMP, jaxify: bool = False,
                 hairpin: int = DEFAULT_HAIRPIN, seed: int = 1):

        self.seed = seed
        if jaxify:
            self.hash = get_jax_float_hash_fn(self.seed)
        else:
            self.hash = lambda *args: float_hash(self.seed, *args)

        self.special_hairpins = DEFAULT_SPECIAL_HAIRPINS
        self.special_hairpin_energies = [self.hash(sp_hrpn_idx, 11) for sp_hrpn_idx in range(len(self.special_hairpins))]

        self.jaxify = jaxify
        self.process_special_hairpins()

        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin


    def hash(self, *args):
        return float_hash(self.seed, *args)

    def en_ext_branch(self, bi, bj):
        return self.hash(bi, bj, 1)

    def en_multi_branch(self, bi, bk):
        return self.hash(bi, bk, 2)

    def en_multi_closing(self, bi, bj):
        return self.hash(bi, bj, 9)

    def en_multi_unpaired(self):
        return self.hash(1)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return self.hash(bi, bj, bip1, bjm1, nunpaired, 10)

    def en_hairpin_special(self, id):
        return self.hash(id, 11)

    def en_stack(self, bi, bj, bk, bl):
        return self.hash(bi, bj, bk, bl, 12)

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return self.hash(bi, bj, bk, bl, nunpaired, 13)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return self.hash(bi, bj, bip1, bjm1, 14)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return self.hash(bi, bj, bim1, bjp1, 15)

    def en_internal_init(self, sz):
        return self.hash(sz, 16)

    def en_internal_asym(self, lup, rup):
        return self.hash(lup, rup, 17)


class StandardNNModel(Model):
    """
    The nearest neighbor energy model without CTDs (i.e. `d0` in ViennaRNA).

    Attributes:
      param_path: A file path to nearest neighbor parameters.
      max_precompute: The maximum sequence length to precompute for initiation costs and asymmetry costs.
      temp: The temperature in Kelvin
      hairpin: The minimum hairpin size.
    """

    def __init__(self, params_path=TURNER_2004,
                 max_precompute=MAX_PRECOMPUTE, temp=CELL_TEMP,
                 hairpin=DEFAULT_HAIRPIN
    ):
        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin

        self.max_precompute = max_precompute
        self.nn_params = NNParams(params_path, max_precompute=max_precompute,
                                  postprocess=False, save_sp_hairpins_jax=False,
                                  temp=self.temp)

        self.jaxify = False
        self.special_hairpins = self.nn_params.special_hairpins
        self.special_hairpin_energies = self.nn_params.special_hairpin_energies
        self.process_special_hairpins()

    def en_ext_branch(self, bi, bj):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0

        en = 0.0
        if pair in NON_GC_PAIRS:
            en += self.nn_params.params['non_gc_closing_penalty']
        return boltz_onp(en, t=self.temp)

    def en_multi_branch(self, bi, bk):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bk]
        if pair not in ALL_PAIRS:
            return 0.0

        dg = self.nn_params.params['ml_branch']
        if pair in NON_GC_PAIRS:
            dg += self.nn_params.params['non_gc_closing_penalty']

        return boltz_onp(dg, t=self.temp)

    def en_multi_unpaired(self):
        return boltz_onp(self.nn_params.params['ml_unpaired'], t=self.temp)

    def _en_term_mismatch(self, bim1, bi, bj, bjp1):

        mm_table = self.nn_params.params['mismatch_multi']
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in ALL_PAIRS:
            return 0.0

        b_dangle5 = bim1
        b_dangle3 = bjp1
        if b_dangle5 != INVALID_BASE and b_dangle3 != INVALID_BASE:
            en = mm_table[pair][RNA_ALPHA[b_dangle5] + RNA_ALPHA[b_dangle3]]
        else:
            en = 0.0
        return en


    def en_multi_closing(self, bi, bj):
        closing_pair_dg = self.nn_params.params['ml_initiation']
        closing_pair_dg += self.nn_params.params['ml_branch']
        closing_pair_dg += non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bj] * boltz_onp(closing_pair_dg, t=self.temp)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        initiation = self.nn_params.params['hairpin'][nunpaired] # FIXME: check against MAX_LOOP

        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if nunpaired == 3:
            non_gc_closing_penalty = 0
            if RNA_ALPHA[bi] + RNA_ALPHA[bj] in NON_GC_PAIRS:
                non_gc_closing_penalty = self.nn_params.params['non_gc_closing_penalty']
            en = initiation + non_gc_closing_penalty
        else:
            mismatch = self.nn_params.params['mismatch_hairpin'][pair][RNA_ALPHA[bip1]+RNA_ALPHA[bjm1]]
            en = initiation + mismatch

        return boltz_onp(en, t=self.temp)

    def en_hairpin_special(self, id):
        # id is the index into self.special_hairpins
        en = self.nn_params.special_hairpin_energies[id]
        return boltz_onp(en, t=self.temp)

    def _en_stack(self, bi, bj, bk, bl):
        pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
        en = self.nn_params.params['stack'][pair1][pair2] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_onp(self._en_stack(bi, bj, bk, bl), t=self.temp)

    def bulge_initiation(self, u):
        return self.nn_params.params['bulge'][u] # FIXME: check against MAX_LOOP

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)
        if nunpaired == 1:
            bulge_dg += self._en_stack(bi, bj, bk, bl)
        else:
            pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
            if pair1 in NON_GC_PAIRS:
                bulge_dg += self.nn_params.params['non_gc_closing_penalty']

            pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
            if pair2 in NON_GC_PAIRS:
                bulge_dg += self.nn_params.params['non_gc_closing_penalty']

        return boltz_onp(bulge_dg, t=self.temp)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        en = self.nn_params.params['mismatch_interior'][pair][RNA_ALPHA[bip1] + RNA_ALPHA[bjm1]]
        return boltz_onp(en, t=self.temp)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        pair = RNA_ALPHA[bj] + RNA_ALPHA[bi]
        return boltz_onp(self.nn_params.params['mismatch_interior'][pair][RNA_ALPHA[bjp1] + RNA_ALPHA[bim1]], t=self.temp)

    def _en_internal_init(self, sz):
        return self.nn_params.params['interior'][sz] # FIXME: check against MAX_LOOP
    def en_internal_init(self, sz):
        return boltz_onp(self._en_internal_init(sz), t=self.temp)

    def _en_internal_asym(self, lup, rup):
        asym = onp.abs(lup-rup)
        dg = self.nn_params.params['asymmetry'] * asym
        return onp.min([self.nn_params.params['asymmetry_max'], dg])
    def en_internal_asym(self, lup, rup):
        return boltz_onp(self._en_internal_asym(lup, rup), t=self.temp)

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
        bip1 = RNA_ALPHA[bip1]
        bjm1 = RNA_ALPHA[bjm1]
        bkm1 = RNA_ALPHA[bkm1]
        blp1 = RNA_ALPHA[blp1]

        if lup == 1 and rup == 1:
            int11_dg = self.nn_params.params['int11'][pair1][pair2][bip1][bjm1]
            return boltz_onp(int11_dg, t=self.temp)
        elif lup == 1 and rup == 2:
            int12_dg = self.nn_params.params['int21'][pair1][pair2][bip1+blp1][bjm1]
            return boltz_onp(int12_dg, t=self.temp)
        elif lup == 2 and rup == 1:
            int21_dg = self.nn_params.params['int21'][pair2][pair1][blp1+ bip1][bkm1]
            return boltz_onp(int21_dg, t=self.temp)
        elif lup == 2 and rup == 2:
            int22_dg = self.nn_params.params['int22'][pair1][pair2][bip1+bkm1][blp1+bjm1]
            return boltz_onp(int22_dg, t=self.temp)

        mm_table = self.nn_params.params['mismatch_interior']
        if lup == 1 or rup == 1:
            mm_table = self.nn_params.params['mismatch_interior_1n']
        elif (lup == 2 and rup == 3) or (lup == 3 and rup == 2):
            mm_table = self.nn_params.params['mismatch_interior_23']

        en = self.en_internal_init(lup+rup)*self.en_internal_asym(lup, rup) \
             * boltz_onp(mm_table[pair1][bip1+bjm1], t=self.temp) \
             * boltz_onp(mm_table[pair2][blp1+bkm1], t=self.temp)
        return en



class JaxNNModel(Model):
    """
    A JIT-compatible version of the nearest neighbor energy model without CTDs (i.e. `d0` in ViennaRNA).

    Attributes:
      param_path: A file path to nearest neighbor parameters.
      max_precompute: The maximum sequence length to precompute for initiation costs and asymmetry costs.
      temp: The temperature in Kelvin
      hairpin: The minimum hairpin size.
    """

    def __init__(self, params_path=TURNER_2004,
                 max_precompute=MAX_PRECOMPUTE, temp=CELL_TEMP,
                 hairpin=DEFAULT_HAIRPIN
    ):
        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin

        self.max_precompute = max_precompute
        self.nn_params = NNParams(params_path, max_precompute=max_precompute,
                                  postprocess=True, save_sp_hairpins_jax=True,
                                  temp=self.temp)

        self.jaxify = True
        self.special_hairpins = self.nn_params.special_hairpins
        self.special_hairpin_energies = self.nn_params.special_hairpin_energies
        self.process_special_hairpins()

    def en_ext_branch(self, bi, bj):
        en = non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']
        return all_pairs_mat[bi, bj] * boltz_jnp(en, t=self.temp)

    def en_multi_branch(self, bi, bk):
        dg = self.nn_params.params['ml_branch']
        dg += non_gc_pairs_mat[bi, bk] * self.nn_params.params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bk] * boltz_jnp(dg, t=self.temp)

    def en_multi_unpaired(self):
        return boltz_jnp(self.nn_params.params['ml_unpaired'], t=self.temp)


    def _en_term_mismatch(self, bim1, bi, bj, bjp1):
        mm_table = self.nn_params.params['mismatch_multi']
        b_dangle5 = bim1
        b_dangle3 = bjp1
        both_dangles_cond = (b_dangle5 != INVALID_BASE) & (b_dangle3 != INVALID_BASE)
        en = jnp.where(both_dangles_cond, mm_table[bi, bj, b_dangle5, b_dangle3], 0.0)
        return en


    def en_multi_closing(self, bi, bj):
        closing_pair_dg = self.nn_params.params['ml_initiation']
        closing_pair_dg += self.nn_params.params['ml_branch']
        closing_pair_dg += non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bj] * boltz_jnp(closing_pair_dg, t=self.temp)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        initiation = self.nn_params.params['hairpin'][nunpaired] # Note: nunpaired must be less than MAX_PRECOMPUTE

        # only used if u != 3
        mismatch = self.nn_params.params['mismatch_hairpin'][bi, bj, bip1, bjm1]

        # only used if u == 3
        non_gc_closing_penalty = non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']

        en = jnp.where(nunpaired == 3, initiation + non_gc_closing_penalty, initiation + mismatch)
        return boltz_jnp(en, t=self.temp)

    def en_hairpin_special(self, id):
        # id is the index into self.special_hairpins
        en = self.nn_params.special_hairpin_energies[id]
        return boltz_jnp(en, t=self.temp)

    def _en_stack(self, bi, bj, bk, bl):
        en = self.nn_params.params['stack'][bi, bj, bl, bk] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_jnp(self._en_stack(bi, bj, bk, bl), t=self.temp)

    def bulge_initiation(self, u):
        return self.nn_params.params['bulge'][u] # Note: u must be less than MAX_PRECOMPUTE

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)

        # Note: only used if nunpaired == 1
        stack_dg = self._en_stack(bi, bj, bk, bl)

        # Note: only used if nunpaired nunpaired != 1
        gc_penalty_dg = non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty'] # FIXME: should these be self.nn_params.params (JAX) even though it's a scalar?
        gc_penalty_dg += non_gc_pairs_mat[bl, bk] * self.nn_params.params['non_gc_closing_penalty']

        bulge_dg += jnp.where(nunpaired == 1, stack_dg, gc_penalty_dg)
        return boltz_jnp(bulge_dg, t=self.temp)

    def _en_il_inner_mismatch(self, bi, bj, bip1, bjm1, mm_table):
        return boltz_jnp(mm_table[bi, bj, bip1, bjm1], t=self.temp)
    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return self._en_il_inner_mismatch(bi, bj, bip1, bjm1,
                                          mm_table=self.nn_params.params["mismatch_interior"])

    def _en_il_outer_mismatch(self, bi, bj, bim1, bjp1, mm_table):
        return boltz_jnp(mm_table[bj, bi, bjp1, bim1], t=self.temp)
    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return self._en_il_outer_mismatch(bi, bj, bim1, bjp1,
                                          mm_table=self.nn_params.params["mismatch_interior"])

    def _en_internal_init(self, sz):
        return self.nn_params.params['interior'][sz] # Note: sz must be less than MAX_PRECOMPUTE
    def en_internal_init(self, sz):
        return boltz_jnp(self._en_internal_init(sz), t=self.temp)

    def _en_internal_asym(self, lup, rup):
        return self.nn_params.params['asymmetry_matrix'][lup, rup]
    def en_internal_asym(self, lup, rup):
        return boltz_jnp(self._en_internal_asym(lup, rup), t=self.temp)

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        mm_table = jnp.where((lup == 1) | (rup == 1),
                             self.nn_params.params['mismatch_interior_1n'],
                             jnp.where(((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2)),
                                       self.nn_params.params['mismatch_interior_23'],
                                       self.nn_params.params['mismatch_interior']))

        gen_int = self.en_internal_init(lup+rup) * self.en_internal_asym(lup, rup) \
                  * self._en_il_inner_mismatch(bi, bj, bip1, bjm1, mm_table) \
                  * self._en_il_outer_mismatch(bk, bl, bkm1, blp1, mm_table)

        dg_boltz = jnp.where((lup == 1) & (rup == 1),
                             boltz_jnp(self.nn_params.params['int11'][bi, bj, bl, bk, bip1, bjm1], t=self.temp),
                       jnp.where((lup == 1) & (rup == 2),
                                 boltz_jnp(self.nn_params.params['int21'][bi, bj, bl, bk, bip1, blp1, bjm1], t=self.temp),
                                 jnp.where((lup == 2) & (rup == 1),
                                           boltz_jnp(self.nn_params.params['int21'][bl, bk, bi, bj, blp1, bip1, bkm1], t=self.temp),
                                           jnp.where((lup == 2) & (rup == 2),
                                                     boltz_jnp(self.nn_params.params['int22'][bi, bj, bl, bk, bip1, bkm1, blp1, bjm1], t=self.temp),
                                                     gen_int))))

        return dg_boltz


@dataclasses.dataclass
class JaxNNModel2():

    special_hairpin_lens: Array
    special_hairpin_idxs: Array
    special_hairpin_start_pos: Array
    special_hairpin_energies: Array
    n_special_hairpins: int = dataclasses.static_field()
    thermo_params: Dict

    temp: float = dataclasses.static_field()
    beta: float = dataclasses.static_field()
    hairpin: int = dataclasses.static_field()


    def en_ext_branch(self, bi, bj):
        en = non_gc_pairs_mat[bi, bj] * self.thermo_params['non_gc_closing_penalty']
        return all_pairs_mat[bi, bj] * boltz_jnp(en, t=self.temp)

    def en_multi_branch(self, bi, bk):
        dg = self.thermo_params['ml_branch']
        dg += non_gc_pairs_mat[bi, bk] * self.thermo_params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bk] * boltz_jnp(dg, t=self.temp)

    def en_multi_unpaired(self):
        return boltz_jnp(self.thermo_params['ml_unpaired'], t=self.temp)

    def _en_term_mismatch(self, bim1, bi, bj, bjp1):
        mm_table = self.thermo_params['mismatch_multi']
        b_dangle5 = bim1
        b_dangle3 = bjp1
        both_dangles_cond = (b_dangle5 != INVALID_BASE) & (b_dangle3 != INVALID_BASE)
        en = jnp.where(both_dangles_cond, mm_table[bi, bj, b_dangle5, b_dangle3], 0.0)
        return en


    def en_multi_closing(self, bi, bj):
        closing_pair_dg = self.thermo_params['ml_initiation']
        closing_pair_dg += self.thermo_params['ml_branch']
        closing_pair_dg += non_gc_pairs_mat[bi, bj] * self.thermo_params['non_gc_closing_penalty']

        return all_pairs_mat[bi, bj] * boltz_jnp(closing_pair_dg, t=self.temp)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        initiation = self.thermo_params['hairpin'][nunpaired] # Note: nunpaired must be less than MAX_PRECOMPUTE

        # only used if u != 3
        mismatch = self.thermo_params['mismatch_hairpin'][bi, bj, bip1, bjm1]

        # only used if u == 3
        non_gc_closing_penalty = non_gc_pairs_mat[bi, bj] * self.thermo_params['non_gc_closing_penalty']

        en = jnp.where(nunpaired == 3, initiation + non_gc_closing_penalty, initiation + mismatch)
        return boltz_jnp(en, t=self.temp)

    def en_hairpin_special(self, id):
        # id is the index into self.special_hairpins
        en = self.special_hairpin_energies[id]
        return boltz_jnp(en, t=self.temp)

    def _en_stack(self, bi, bj, bk, bl):
        en = self.thermo_params['stack'][bi, bj, bl, bk] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_jnp(self._en_stack(bi, bj, bk, bl), t=self.temp)

    def bulge_initiation(self, u):
        return self.thermo_params['bulge'][u] # Note: u must be less than MAX_PRECOMPUTE

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)

        # Note: only used if nunpaired == 1
        stack_dg = self._en_stack(bi, bj, bk, bl)

        # Note: only used if nunpaired nunpaired != 1
        gc_penalty_dg = non_gc_pairs_mat[bi, bj] * self.thermo_params['non_gc_closing_penalty'] # FIXME: should these be self.thermo_params (JAX) even though it's a scalar?
        gc_penalty_dg += non_gc_pairs_mat[bl, bk] * self.thermo_params['non_gc_closing_penalty']

        bulge_dg += jnp.where(nunpaired == 1, stack_dg, gc_penalty_dg)
        return boltz_jnp(bulge_dg, t=self.temp)

    def _en_il_inner_mismatch(self, bi, bj, bip1, bjm1, mm_table):
        return boltz_jnp(mm_table[bi, bj, bip1, bjm1], t=self.temp)
    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return self._en_il_inner_mismatch(bi, bj, bip1, bjm1,
                                          mm_table=self.thermo_params["mismatch_interior"])

    def _en_il_outer_mismatch(self, bi, bj, bim1, bjp1, mm_table):
        return boltz_jnp(mm_table[bj, bi, bjp1, bim1], t=self.temp)
    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return self._en_il_outer_mismatch(bi, bj, bim1, bjp1,
                                          mm_table=self.thermo_params["mismatch_interior"])

    def _en_internal_init(self, sz):
        return self.thermo_params['interior'][sz] # Note: sz must be less than MAX_PRECOMPUTE
    def en_internal_init(self, sz):
        return boltz_jnp(self._en_internal_init(sz), t=self.temp)

    def _en_internal_asym(self, lup, rup):
        return self.thermo_params['asymmetry_matrix'][lup, rup]
    def en_internal_asym(self, lup, rup):
        return boltz_jnp(self._en_internal_asym(lup, rup), t=self.temp)

    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        mm_table = jnp.where((lup == 1) | (rup == 1),
                             self.thermo_params['mismatch_interior_1n'],
                             jnp.where(((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2)),
                                       self.thermo_params['mismatch_interior_23'],
                                       self.thermo_params['mismatch_interior']))

        gen_int = self.en_internal_init(lup+rup) * self.en_internal_asym(lup, rup) \
                  * self._en_il_inner_mismatch(bi, bj, bip1, bjm1, mm_table) \
                  * self._en_il_outer_mismatch(bk, bl, bkm1, blp1, mm_table)

        dg_boltz = jnp.where((lup == 1) & (rup == 1),
                             boltz_jnp(self.thermo_params['int11'][bi, bj, bl, bk, bip1, bjm1], t=self.temp),
                       jnp.where((lup == 1) & (rup == 2),
                                 boltz_jnp(self.thermo_params['int21'][bi, bj, bl, bk, bip1, blp1, bjm1], t=self.temp),
                                 jnp.where((lup == 2) & (rup == 1),
                                           boltz_jnp(self.thermo_params['int21'][bl, bk, bi, bj, blp1, bip1, bkm1], t=self.temp),
                                           jnp.where((lup == 2) & (rup == 2),
                                                     boltz_jnp(self.thermo_params['int22'][bi, bj, bl, bk, bip1, bkm1, blp1, bjm1], t=self.temp),
                                                     gen_int))))

        return dg_boltz



def calculate(str_seq, db, em: Model):
    seq = [RNA_ALPHA.index(c) for c in str_seq]

    ch, right = structure_tree(db)

    def calc_rec(atl):
        if atl == -1:
            sm = 1
            for cl in ch[atl]:
                sm *= calc_rec(cl)*em.en_ext_branch(seq[cl], seq[right[cl]])
            return sm
        if atl not in ch:
            s = str_seq[atl:right[atl]+1]
            idx = em.special_hairpins.index(s) if s in em.special_hairpins else -1
            return em.en_hairpin_special(idx) if idx != -1 else em.en_hairpin_not_special(
                seq[atl], seq[right[atl]], seq[atl+1], seq[right[atl]-1], right[atl]-atl-1)
        elif len(ch[atl]) == 1:
            cl, cr = ch[atl][0], right[ch[atl][0]]
            if cl == atl+1 and cr == right[atl]-1:
                return em.en_stack(seq[atl], seq[right[atl]], seq[cl], seq[cr])*calc_rec(cl)
            elif cl == atl+1 or cr == right[atl]-1:
                nunpaired = max(cl-atl-1, right[atl]-cr-1)
                return em.en_bulge(seq[atl], seq[right[atl]], seq[cl], seq[cr], nunpaired)*calc_rec(cl)
            else:
                bi = seq[atl]
                bj = seq[right[atl]]
                bip1 = seq[atl+1]
                bjm1 = seq[right[atl]-1]
                bk = seq[cl]
                bl = seq[cr]
                bkm1 = seq[cl-1]
                blp1 = seq[cr+1]
                lup = cl-atl-1
                rup = right[atl]-cr-1
                return em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*calc_rec(cl)
        else:
            sm = em.en_multi_closing(seq[atl], seq[right[atl]])
            for cl in ch[atl]:
                sm *= calc_rec(cl)*em.en_multi_branch(seq[cl], seq[right[cl]])
            return sm
    return calc_rec(-1)
