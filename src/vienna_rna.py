import pdb
import numpy as np
import RNA

from utils import R, CELL_TEMP, boltz_onp


# https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/examples_python.html
def vienna_energy(seq, struct):
    fc = RNA.fold_compound(seq)

    (mfe_struct, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    # (pp, pf) = fc.pf()
    # (centroid_struct, dist) = fc.centroid()
    # centroid_en = fc.eval_structure(centroid_struct)

    struct_en = fc.eval_structure(struct)
    # NOTE: RNA.eval_structure_simple(seq, structure, verbose=1). Can also run Marco's code with --verbose flag.
    return struct_en

def get_vienna_pf(seq):
    md = RNA.md();
    md.uniq_ML = 1
    md.dangles = 2
    md.noLP = False
    fc = RNA.fold_compound(seq, md)

    # compute MFE and MFE structure
    (mfe_struct, mfe) = fc.mfe()

    # rescale Boltzmann factors for partition function computation
    fc.exp_params_rescale(mfe)

    (_, g) = fc.pf() # get the ensemble free energy
    pf = np.exp(g / (-R * CELL_TEMP))  # convert the ensemble free energy to the partition function
    return pf



def vienna_prob(seq, struct):
    dg = vienna_energy(seq, struct)
    dg_boltz = boltz_onp(dg)
    pf = get_vienna_pf(seq)
    return dg_boltz / pf

def subopt(fc, energy_delta=1e6):
    # set energy_delta to ~inf to get all structures
    # sub = fc.subopt(int(energy_delta*100))
    sub = fc.subopt(int(energy_delta))
    return [s.structure for s in sub]
