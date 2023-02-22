from jax_rnafold.common import nussinov, read_vienna_params

if __name__ == "__main__":
    params_dir = "misc"
    params_fname = "rna_turner2004.par"
    params_path = f"{params_dir}/{params_fname}"
    params_2004 = read_vienna_params.read(params_path)
