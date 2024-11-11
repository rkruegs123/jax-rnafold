import numpy as onp
from tqdm import tqdm
import pdb
from abc import ABC, abstractmethod
import random

import jax
import jax.numpy as jnp

from jax_rnafold.common.utils import RNA_ALPHA, INVALID_BASE, RNA_ALPHA_IDX, VALID_PAIRS, DEFAULT_HAIRPIN
from jax_rnafold.common.utils import boltz_onp, boltz_jnp
from jax_rnafold.common.utils import MAX_LOOP, NON_GC_PAIRS, kb, CELL_TEMP, MAX_PRECOMPUTE, TURNER_2004
from jax_rnafold.common.utils import all_pairs_mat, non_gc_pairs_mat
from jax_rnafold.common import utils
from jax_rnafold.common.energy_hash import float_hash, get_jax_float_hash_fn
from jax_rnafold.common.read_vienna_params import NNParams


jax.config.update("jax_enable_x64", True)


DEFAULT_SPECIAL_HAIRPINS = ["AAA", "AAAA", "AAAAAA"]
allowed_sp_hairpin_lens = set([3, 4, 6])
for sp_hairpin in DEFAULT_SPECIAL_HAIRPINS:
    len_hairpin = len(sp_hairpin)
    assert(len_hairpin in allowed_sp_hairpin_lens)

class Model(ABC):
    temp: float # Kelvin
    beta: float
    hairpin: int

    jaxify: bool

    special_hairpins: list[str]
    special_hairpin_energies: list[float]

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
    def en_ext_branch(self, bim1, bi, bj, bjp1):
        pass

    @abstractmethod
    def en_multi_branch(self, bim1, bi, bk, bkp1):
        pass

    @abstractmethod
    def en_multi_closing(self, bi, bip1, bjm1, bj):
        pass

    @abstractmethod
    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        # Note that vienna ignores the all-C case
        pass

    @abstractmethod
    def en_hairpin_special(self, id):
        # id is the index into self.special_hairpins
        pass

    @abstractmethod
    def en_stack(self, bi, bj, bk, bl):
        pass

    @abstractmethod
    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        pass

    @abstractmethod
    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        pass

    @abstractmethod
    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        pass

    @abstractmethod
    def en_internal_init(self, sz):
        pass

    @abstractmethod
    def en_internal_asym(self, lup, rup):
        pass

    # Note: abs not differentiable
    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):
        en = self.en_internal_init(lup+rup)*self.en_internal_asym(lup, rup)*self.en_il_inner_mismatch(
            bi, bj, bip1, bjm1)*self.en_il_outer_mismatch(bk, bl, bkm1, blp1)
        return en


class All1Model(Model):
    def __init__(self, temp=CELL_TEMP, jaxify=False, hairpin=DEFAULT_HAIRPIN):
        self.special_hairpins = DEFAULT_SPECIAL_HAIRPINS
        self.special_hairpin_energies = [0.0 for _ in self.special_hairpins]

        self.jaxify = jaxify
        self.process_special_hairpins()

        self.temp = temp
        self.beta = 1 / (kb*self.temp)

        self.hairpin = hairpin

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        return 1

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        return 1

    def en_multi_closing(self, bi, bip1, bjm1, bj):
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
    def __init__(self, temp=CELL_TEMP, jaxify=False, hairpin=DEFAULT_HAIRPIN, seed=1):
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

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        return float_hash(self.seed, bim1, bi, bj, bjp1, 1)

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        return float_hash(self.seed, bim1, bi, bk, bkp1, 2)

    def en_multi_closing(self, bi, bip1, bjm1, bj):
        return float_hash(self.seed, bi, bip1, bjm1, bj, 3)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return float_hash(self.seed, bi, bj, bip1, bjm1, nunpaired, 4)

    def en_hairpin_special(self, id):
        return float_hash(self.seed, id, 5)

    def en_stack(self, bi, bj, bk, bl):
        return float_hash(self.seed, bi, bj, bk, bl, 6)

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return float_hash(self.seed, bi, bj, bk, bl, nunpaired, 7)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return float_hash(self.seed, bi, bj, bip1, bjm1, 8)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return float_hash(self.seed, bi, bj, bim1, bjp1, 9)

    def en_internal_init(self, sz):
        return float_hash(self.seed, sz, 10)

    def en_internal_asym(self, lup, rup):
        return float_hash(self.seed, lup, rup, 11)


class RandomMultiloopModel(RandomModel):

    def en_ext_branch(self, bim1, bi, bj, bjp1):
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


class RandomBulgeModel(RandomModel):

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        return 1

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        return 1

    def en_multi_closing(self, bi, bip1, bjm1, bj):
        return 1

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 1

    def en_hairpin_special(self, id):
        return 1

    def en_stack(self, bi, bj, bk, bl):
        return 1

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return 1

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return 1

    def en_internal_init(self, sz):
        return 1

    def en_internal_asym(self, lup, rup):
        return 1


class RandomILModel(RandomModel):

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        return 1

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        return 1

    def en_multi_closing(self, bi, bip1, bjm1, bj):
        return 1

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        return 1

    def en_hairpin_special(self, id):
        return 1

    def en_stack(self, bi, bj, bk, bl):
        return 1

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        return 1


class RandomHairpinModel(RandomModel):

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        return 1

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        return 1

    def en_multi_closing(self, bi, bip1, bjm1, bj):
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


class StandardNNModel(Model):
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

    def score_stem_extloop_multiloop(self, bi, bj, b_dangle5, b_dangle3, mm_table):
        dg = 0
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]

        if pair in NON_GC_PAIRS:
            dg += self.nn_params.params['non_gc_closing_penalty']
        if b_dangle5 != INVALID_BASE and b_dangle3 != INVALID_BASE:
            dg += mm_table[pair][RNA_ALPHA[b_dangle5] + RNA_ALPHA[b_dangle3]]
        else:
            if b_dangle5 != INVALID_BASE:
                dg += self.nn_params.params['dangle5'][pair][RNA_ALPHA[b_dangle5]]
            if b_dangle3 != INVALID_BASE:
                dg += self.nn_params.params['dangle3'][pair][RNA_ALPHA[b_dangle3]]
        return dg

    # Helper that returns the energy rather than the boltzmann weight
    def _en_stack(self, bi, bj, bk, bl):
        pair1 = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        pair2 = RNA_ALPHA[bl] + RNA_ALPHA[bk]
        en = self.nn_params.params['stack'][pair1][pair2] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_onp(self._en_stack(bi, bj, bk, bl), t=self.temp)


    def en_hairpin_special(self, id):
        en = self.nn_params.special_hairpin_energies[id]
        return boltz_onp(en, t=self.temp)

    def get_max_loop_correction(self, u):
        return onp.floor(1.07856 * onp.log(u/MAX_LOOP) * 100) / 100

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        bip1 = RNA_ALPHA[bip1]
        bjm1 = RNA_ALPHA[bjm1]

        if nunpaired < MAX_LOOP:
            initiation = self.nn_params.params['hairpin'][nunpaired]
        else:
            initiation = self.nn_params.params['hairpin'][MAX_LOOP] \
                         + self.get_max_loop_correction(nunpaired)

        if nunpaired == 3:
            non_gc_closing_penalty = 0
            if RNA_ALPHA[bi] + RNA_ALPHA[bj] in NON_GC_PAIRS:
                non_gc_closing_penalty = self.nn_params.params['non_gc_closing_penalty']
            en = initiation + non_gc_closing_penalty
        else:
            mismatch = self.nn_params.params['mismatch_hairpin'][pair][bip1+bjm1]
            en = initiation + mismatch

        return boltz_onp(en, t=self.temp)


    def en_ext_branch(self, bim1, bi, bj, bjp1):
        # note: we don't have to check conditions for INVALID_BASE as we handle this in the recursions
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in VALID_PAIRS:
            return 0.0

        b_dangle5 = bim1
        b_dangle3 = bjp1
        en = self.score_stem_extloop_multiloop(bi, bj, b_dangle5, b_dangle3,
                                               self.nn_params.params['mismatch_exterior'])
        return boltz_onp(en, t=self.temp)


    # Helper that returns the energy rather than the boltzmann weight
    def _en_internal_init(self, sz):
        # Note: sz is the number of unpaired
        if sz < MAX_LOOP:
            return self.nn_params.params['interior'][sz]
        else:
            return self.nn_params.params['interior'][MAX_LOOP] + self.get_max_loop_correction(sz)
    def en_internal_init(self, sz):
        return boltz_onp(self._en_internal_init(sz), t=self.temp)

    # Helper that returns the energy rather than the boltzmann weight
    def _en_internal_asym(self, lup, rup):
        asym = onp.abs(lup-rup)
        dg = self.nn_params.params['asymmetry'] * asym
        return onp.min([self.nn_params.params['asymmetry_max'], dg])

    def en_internal_asym(self, lup, rup):
        return boltz_onp(self._en_internal_asym(lup, rup), t=self.temp)


    # Note: neither `en_il_inner_mismatch` or `en_il_outer_mismatch` will be used with the NNModel,
    # but including for completeness.
    # Note that they will only be called for the general case, so we only lookup in the
    # `mismatch_interior` table
    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        bip1 = RNA_ALPHA[bip1]
        bjm1 = RNA_ALPHA[bjm1]
        return boltz_onp(self.nn_params.params['mismatch_interior'][pair][bip1+bjm1], t=self.temp)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        pair_rev = RNA_ALPHA[bj] + RNA_ALPHA[bi]
        bim1 = RNA_ALPHA[bim1]
        bjp1 = RNA_ALPHA[bjp1]
        return boltz_onp(self.nn_params.params['mismatch_interior'][pair_rev][bjp1+bim1], t=self.temp)

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

        init_dg = self._en_internal_init(lup + rup)
        asymmetry_dg = self._en_internal_asym(lup, rup) # Note: could also use the lookup table

        mismatch_dg = mm_table[pair1][bip1+bjm1]
        mismatch_dg += mm_table[pair2][blp1+bkm1]

        int_dg = init_dg + asymmetry_dg + mismatch_dg
        return boltz_onp(int_dg, t=self.temp)


    def bulge_initiation(self, u):
        if u < MAX_LOOP:
            return self.nn_params.params['bulge'][u]
        else:
            return self.nn_params.params['bulge'][MAX_LOOP] + self.get_max_loop_correction(u)

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


    def en_multi_branch(self, bim1, bi, bk, bkp1):

        pair = RNA_ALPHA[bi] + RNA_ALPHA[bk]
        if pair not in VALID_PAIRS:
            return 0.0

        dg = self.nn_params.params['ml_branch']

        b_dangle5 = bim1
        b_dangle3 = bkp1

        dg += self.score_stem_extloop_multiloop(bi, bk, b_dangle5, b_dangle3,
                                                self.nn_params.params['mismatch_multi'])
        return boltz_onp(dg, t=self.temp)


    def en_multi_closing(self, bi, bip1, bjm1, bj):

        pair = RNA_ALPHA[bi] + RNA_ALPHA[bj]
        if pair not in VALID_PAIRS:
            return 0.0

        closing_pair_dg = self.nn_params.params['ml_initiation']
        closing_pair_dg += self.nn_params.params['ml_branch']

        # note: we don't have to check conditions for INVALID_BASE as we handle this in the recursions
        # note: we swap dangle3 and dangle5 here (and i and j effectively)
        b_dangle3 = bip1 # note these are swapped
        b_dangle5 = bjm1 # note these are swapped

        closing_pair_dg += self.score_stem_extloop_multiloop(bj, bi, b_dangle5, b_dangle3,
                                                             self.nn_params.params['mismatch_multi'])

        return boltz_onp(closing_pair_dg, t=self.temp)

class JaxNNModel(Model):
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

    def _en_stack(self, bi, bj, bk, bl):
        en = self.nn_params.params['stack'][bi, bj, bl, bk] # swap bk and bl
        return en
    def en_stack(self, bi, bj, bk, bl):
        return boltz_jnp(self._en_stack(bi, bj, bk, bl), t=self.temp)

    def en_il_inner_mismatch(self, bi, bj, bip1, bjm1):
        return boltz_jnp(self.nn_params.params['mismatch_interior'][bi, bj, bip1, bjm1], t=self.temp)

    def en_il_outer_mismatch(self, bi, bj, bim1, bjp1):
        return boltz_jnp(self.nn_params.params['mismatch_interior'][bj, bi, bjp1, bim1], t=self.temp)

    def bulge_initiation(self, u):
        return self.nn_params.params['bulge'][u] # Note: u must be less than MAX_PRECOMPUTE

    # Helper that returns the energy rather than the boltzmann weight
    def _en_internal_init(self, sz):
        return self.nn_params.params['interior'][sz] # Note: sz must be less than MAX_PRECOMPUTE
    def en_internal_init(self, sz):
        return boltz_jnp(self._en_internal_init(sz), t=self.temp)

    def en_hairpin_not_special(self, bi, bj, bip1, bjm1, nunpaired):

        initiation = self.nn_params.params['hairpin'][nunpaired] # Note: nunpaired must be less than MAX_PRECOMPUTE

        # only used if u != 3
        mismatch = self.nn_params.params['mismatch_hairpin'][bi, bj, bip1, bjm1]

        # only used if u == 3
        non_gc_closing_penalty = non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']

        en = jnp.where(nunpaired == 3, initiation + non_gc_closing_penalty, initiation + mismatch)
        return boltz_jnp(en, t=self.temp)

    def en_hairpin_special(self, id):
        en = self.nn_params.special_hairpin_energies[id]
        return boltz_jnp(en, t=self.temp)

    def en_bulge(self, bi, bj, bk, bl, nunpaired):
        bulge_dg = self.bulge_initiation(nunpaired)

        # Note: only used if nunpaired == 1
        stack_dg = self._en_stack(bi, bj, bk, bl)

        # Note: only used if nunpaired nunpaired != 1
        gc_penalty_dg = non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']
        gc_penalty_dg += non_gc_pairs_mat[bl, bk] * self.nn_params.params['non_gc_closing_penalty']

        bulge_dg += jnp.where(nunpaired == 1, stack_dg, gc_penalty_dg)
        return boltz_jnp(bulge_dg, t=self.temp)

    def score_stem_extloop_multiloop(self, bi, bj, b_dangle5, b_dangle3, mm_table):
        dg = 0

        # GC penalty
        dg += non_gc_pairs_mat[bi, bj] * self.nn_params.params['non_gc_closing_penalty']

        both_dangles_dg = mm_table[bi, bj, b_dangle5, b_dangle3]

        not_both_dangles_dg = jnp.where(b_dangle5 != INVALID_BASE,
                                        self.nn_params.params['dangle5'][bi, bj, b_dangle5], 0.0)
        not_both_dangles_dg += jnp.where(b_dangle3 != INVALID_BASE,
                                         self.nn_params.params['dangle3'][bi, bj, b_dangle3], 0.0)

        both_dangles_cond = (b_dangle5 != INVALID_BASE) & (b_dangle3 != INVALID_BASE)
        dg += jnp.where(both_dangles_cond, both_dangles_dg, not_both_dangles_dg)
        return dg

    def en_ext_branch(self, bim1, bi, bj, bjp1):
        # note: we don't have to check conditions for INVALID_BASE as we handle this in the recursions
        b_dangle5 = bim1
        b_dangle3 = bjp1
        en = self.score_stem_extloop_multiloop(bi, bj, b_dangle5, b_dangle3,
                                               self.nn_params.params['mismatch_exterior'])
        return all_pairs_mat[bi, bj] * boltz_jnp(en, t=self.temp)



    def en_internal(self, bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup):

        # Compute energy for the general case
        mm_table = jnp.where((lup == 1) | (rup == 1),
                             self.nn_params.params['mismatch_interior_1n'],
                             jnp.where(((lup == 2) & (rup == 3)) | ((lup == 3) & (rup == 2)),
                                       self.nn_params.params['mismatch_interior_23'],
                                       self.nn_params.params['mismatch_interior']))
        init_dg = self._en_internal_init(lup + rup)
        asymmetry_dg = self._en_internal_asym(lup, rup)
        mismatch_dg = mm_table[bi, bj, bip1, bjm1]
        mismatch_dg += mm_table[bl, bk, blp1, bkm1]

        general_int_dg = init_dg + asymmetry_dg + mismatch_dg

        en = jnp.where((lup == 1) & (rup == 1),
                       self.nn_params.params['int11'][bi, bj, bl, bk, bip1, bjm1],
                       jnp.where((lup == 1) & (rup == 2),
                                 self.nn_params.params['int21'][bi, bj, bl, bk, bip1, blp1, bjm1],
                                 jnp.where((lup == 2) & (rup == 1),
                                           self.nn_params.params['int21'][bl, bk, bi, bj, blp1, bip1, bkm1],
                                           jnp.where((lup == 2) & (rup == 2),
                                                     self.nn_params.params['int22'][bi, bj, bl, bk, bip1, bkm1, blp1, bjm1],
                                                     general_int_dg))))
        return boltz_jnp(en, t=self.temp)

    def en_multi_closing(self, bi, bip1, bjm1, bj):
        closing_pair_dg = self.nn_params.params['ml_initiation']
        closing_pair_dg += self.nn_params.params['ml_branch']

        # note: we don't have to check conditions for INVALID_BASE as we handle this in the recursions
        # note: we swap dangle3 and dangle5 here (and i and j effectively)
        b_dangle3 = bip1 # note these are swapped
        b_dangle5 = bjm1 # note these are swapped

        closing_pair_dg += self.score_stem_extloop_multiloop(bj, bi, b_dangle5, b_dangle3,
                                                             self.nn_params.params['mismatch_multi'])

        return all_pairs_mat[bi, bj] * boltz_jnp(closing_pair_dg, t=self.temp)

    def en_multi_branch(self, bim1, bi, bk, bkp1):
        dg = self.nn_params.params['ml_branch']

        b_dangle5 = bim1
        b_dangle3 = bkp1

        dg += self.score_stem_extloop_multiloop(bi, bk, b_dangle5, b_dangle3,
                                                self.nn_params.params['mismatch_multi'])

        return all_pairs_mat[bi, bk] * boltz_jnp(dg, t=self.temp)


    # Helper that returns the energy rather than the boltzmann weight
    def _en_internal_asym(self, lup, rup):
        return self.nn_params.params['asymmetry_matrix'][lup, rup]

    def en_internal_asym(self, lup, rup):
        return boltz_jnp(self._en_internal_asym(lup, rup), t=self.temp)



def calculate(seq, db, em: Model):
    n = len(db)

    def b_idx(i):
        if i == -1 or i == n:
            return INVALID_BASE
        return RNA_ALPHA.index(seq[i])
    ch = {-1: []}
    stk = [-1]
    right = [-1]*n
    for i in range(n):
        if db[i] == '(':
            stk.append(i)
        elif db[i] == ')':
            left = stk.pop()
            if stk[-1] not in ch:
                ch[stk[-1]] = []
            ch[stk[-1]].append(left)
            right[left] = i

    def calc_rec(atl):
        if atl == -1:
            sm = 1
            for cl in ch[atl]:
                sm *= calc_rec(cl)*em.en_ext_branch(b_idx(cl-1),
                                                    b_idx(cl), b_idx(right[cl]), b_idx(right[cl]+1))
            return sm
        if atl not in ch:
            s = seq[atl:right[atl]+1]
            idx = em.special_hairpins.index(s) if s in em.special_hairpins else -1
            return em.en_hairpin_special(idx) if idx != -1 else em.en_hairpin_not_special(
                b_idx(atl), b_idx(right[atl]), b_idx(atl+1), b_idx(right[atl]-1), right[atl]-atl-1)
        elif len(ch[atl]) == 1:
            cl, cr = ch[atl][0], right[ch[atl][0]]
            if cl == atl+1 and cr == right[atl]-1:
                return em.en_stack(b_idx(atl), b_idx(right[atl]), b_idx(cl), b_idx(cr))*calc_rec(cl)
            elif cl == atl+1 or cr == right[atl]-1:
                nunpaired = max(cl-atl-1, right[atl]-cr-1)
                return em.en_bulge(b_idx(atl), b_idx(right[atl]), b_idx(cl), b_idx(cr), nunpaired)*calc_rec(cl)
            else:
                bi = b_idx(atl)
                bj = b_idx(right[atl])
                bip1 = b_idx(atl+1)
                bjm1 = b_idx(right[atl]-1)
                bk = b_idx(cl)
                bl = b_idx(cr)
                bkm1 = b_idx(cl-1)
                blp1 = b_idx(cr+1)
                lup = cl-atl-1
                rup = right[atl]-cr-1
                return em.en_internal(bi, bj, bk, bl, bip1, bjm1, bkm1, blp1, lup, rup)*calc_rec(cl)
        else:
            sm = em.en_multi_closing(b_idx(atl), b_idx(
                atl+1), b_idx(right[atl]-1), b_idx(right[atl]))
            for cl in ch[atl]:
                sm *= calc_rec(cl)*em.en_multi_branch(b_idx(cl-1),
                                                      b_idx(cl), b_idx(right[cl]), b_idx(right[cl]+1))
            return sm
    return calc_rec(-1)
