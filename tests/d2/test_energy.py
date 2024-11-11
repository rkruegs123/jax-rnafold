import unittest
import random
from tqdm import tqdm
import pdb
import numpy as onp

from jax_rnafold.common import sampling, vienna_rna, utils
from jax_rnafold.d2.energy import StandardNNModel, calculate


class TestEnergy(unittest.TestCase):
    def fuzz_test(self, n, num_seq, tol_places, max_structs, t_celsius):

        t_kelvin = utils.celsius_to_kelvin(t_celsius)
        em = StandardNNModel(temp=t_kelvin)

        seqs = [utils.get_rand_seq(n) for _ in range(num_seq)]

        for seq in seqs:
            print(f"Sequence: {seq}")
            sampler = sampling.UniformStructureSampler()
            sampler.precomp(seq)
            n_structs = sampler.count_structures()
            if n_structs > max_structs:
                all_structs = [sampler.get_nth(i) for i in random.sample(list(range(n_structs)), max_structs)]
            else:
                all_structs = [sampler.get_nth(i) for i in range(n_structs)]
            all_structs = [utils.matching_to_db(matching) for matching in all_structs]

            print(f"Found {len(all_structs)} structures")
            all_dgs = list()
            vienna_dgs = list()

            vc = vienna_rna.ViennaContext(seq, temp=t_celsius, dangles=2)

            for db_str in tqdm(all_structs):
                print(f"\tStructure: {db_str}")

                matching = utils.db_to_matching(db_str)
                dg_calc = calculate(seq, db_str, em)

                dg_calc = calculate(seq, db_str, em)
                dg = onp.log(dg_calc) / -em.beta
                all_dgs.append(dg)
                print(f"\t\tComputed dG: {dg}")

                vienna_dg = vc.free_energy(db_str)
                vienna_dgs.append(vienna_dg)
                print(f"\t\tViennaRNA dG: {vienna_dg}")

                self.assertAlmostEqual(dg, vienna_dg, places=tol_places)
                print(utils.bcolors.OKGREEN + "\t\tSuccess!\n" + utils.bcolors.ENDC)

    def test_vienna(self):
        # FIXME: fails for t_celsius != 37.0
        n = 35
        num_seq = 5
        max_structs = 100
        self.fuzz_test(n=35, num_seq=5, tol_places=4, max_structs=100, t_celsius=37.0)
