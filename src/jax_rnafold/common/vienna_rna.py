import numpy as np
import pdb
import RNA
import subprocess

from jax_rnafold.common.utils import R, TURNER_2004, TURNER_1999, bcolors, CELL_TEMP
from jax_rnafold.common import utils


class ViennaContext:
    """ A class for computing thermodynamic properties of an RNA sequence via
    ViennaRNA's Python interface.

    Attributes:
      rna: The RNA sequence.
      temp: The temperature in Celsius.
      dangles: The CTD setting for ViennaRNA (e.g. `dangles=0` corresponds to the `d0` option in ViennaRNA).
      noLPs: An option for ignoring lonely pairs.
      params_path: A file path specifying the thermodynamic parameters.
    """
    def __init__(self, rna: str, temp=37.0, dangles=2, noLPs=False,
                 params_path=TURNER_2004):

        self.params_path = params_path
        if self.params_path != TURNER_2004:
            RNA.params_load(self.params_path)


        md = RNA.md()
        md.uniq_ML = 1
        md.dangles = dangles
        self.dangles = dangles
        md.noLP = noLPs
        md.temperature = temp
        self.temp_celsius = temp
        self.temp_kelvin = utils.celsius_to_kelvin(self.temp_celsius)
        self.fc = RNA.fold_compound(rna, md)
        self.pf_computed = False
        self.n = len(rna)

    def __ensure_pf(self):
        if self.pf_computed:
            return
        _, mfe_energy = self.fc.mfe()
        self.fc.exp_params_rescale(mfe_energy)
        self.fc.pf()

        self.pf_computed = True

    def free_energy(self, ss):
        return self.fc.eval_structure(ss)

    def prob(self, ss):
        self.__ensure_pf()
        return self.fc.pr_structure(ss)

    def make_bppt(self) -> list[list[float]]:
        self.__ensure_pf()
        bpp = self.fc.bpp()
        sz = self.fc.length
        res = [[0.0 for _ in range(sz)] for _ in range(sz)]
        for i in range(sz):
            for j in range(sz):
                if j < i:
                    res[i][j] = bpp[j+1][i+1]
                elif i < j:
                    res[i][j] = bpp[i+1][j+1]
        for i in range(sz):
            res[i][i] = 1-sum(res[i])
        return res

    def subopt(self, energy_delta):
        sub = self.fc.subopt(int(energy_delta*100), sorted=0)
        return [s.structure for s in sub]

    def ensemble_defect(self, ss):
        self.__ensure_pf()
        return self.fc.ensemble_defect(ss)

    def mfe(self):
        mfe_struct, mfe_energy = self.fc.mfe()
        return mfe_energy

    def efe(self):
        (_, g) = self.fc.pf()
        return g

    def pf(self):
        if self.dangles != 2 and self.dangles != 0:
            raise RuntimeError(f"RNAlib only defines the partition function for dangles={0,2}")

        self.__ensure_pf()
        (_, g) = self.fc.pf() # get the ensemble free energy
        pf = np.exp(g / (-R * self.temp_kelvin))  # convert the ensemble free energy to the partition function
        return pf

    def psample(self, samples=1, redundant=True):
        self.__ensure_pf()
        return self.fc.pbacktrack(samples, RNA.PBACKTRACK_DEFAULT if redundant else RNA.PBACKTRACK_NON_REDUNDANT)


    def calc_expected_num_unpaired(self):
        bppt = self.make_bppt()
        assert(len(bppt) == self.n)
        return sum([bppt[i][i] for i in range(self.n)])

    def calc_aup(self):
        return self.calc_expected_num_unpaired() / self.n
