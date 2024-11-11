import unittest
import random
from tqdm import tqdm
import pdb
import numpy as onp

from jax_rnafold.common import sampling, vienna_rna, utils
from jax_rnafold.d0.energy import StandardNNModel, calculate

class TestEnergyCalculator(unittest.TestCase):

    def fuzz_test(self, n, max_structs, params_path, tol_places=5):

        em = StandardNNModel(params_path=params_path)
        seq = utils.get_rand_seq(n)
        print(f"Sequence: {seq}")

        sampler = sampling.UniformStructureSampler()
        sampler.precomp(seq)
        n_structs = sampler.count_structures()
        if n_structs > max_structs:
            all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
        else:
            all_structs = [sampler.get_nth(i) for i in range(n_structs)]
        all_structs = [utils.matching_to_db(matching) for matching in all_structs]

        vc = vienna_rna.ViennaContext(seq, utils.kelvin_to_celsius(em.temp), dangles=0, params_path=params_path)

        for db_str in tqdm(all_structs):
            print(f"\tStructure: {db_str}")

            matching = utils.db_to_matching(db_str)
            calc_boltz = calculate(seq, db_str, em)
            calc_dg = onp.log(calc_boltz) * (-1/em.beta)

            vienna_dg = vc.free_energy(db_str)

            print(f"\t\tCalculated dG: {calc_dg}")
            print(f"\t\tVienna dG: {vienna_dg}")
            print(f"\t\tDifference: {onp.abs(vienna_dg - calc_dg)}")

            self.assertAlmostEqual(calc_dg, vienna_dg, places=tol_places)

    def test_vienna(self):
        # self.fuzz_test(n=35, max_structs=250, params_path=utils.TURNER_2004)
        self.fuzz_test(n=15, max_structs=100, params_path=utils.TURNER_2004)
