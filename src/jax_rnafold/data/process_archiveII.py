import pdb
from pathlib import Path
import subprocess
import shutil
from tqdm import tqdm
import os


"""
ArchiveII downloaded from: https://rna.urmc.rochester.edu/pub/archiveII.tar.gz
- under publication 104 at: https://rna.urmc.rochester.edu/publications.html

Processing involves two RNAstructure executables:
- https://rna.urmc.rochester.edu/Text/RemovePseudoknots.html
- https://rna.urmc.rochester.edu/Text/ct2dot.html
"""


rnastructure_dir = Path("/home/ryan/Documents/Harvard/research/brenner/rna-folding/RNAstructure/")
assert(rnastructure_dir.exists())
exe_dir = rnastructure_dir / "exe"
assert(exe_dir.exists())
rem_pseudoknots_exe = exe_dir / "RemovePseudoknots"
ct2dot_exe = exe_dir / "ct2dot"
data_tables_dir = rnastructure_dir / "data_tables"
assert(data_tables_dir.exists())
os.environ["DATAPATH"] = str(data_tables_dir)

unprocessed_dir = Path("archiveII/")
assert(unprocessed_dir.exists())

processed_dir = Path("archiveII_processed/")
processed_dir.mkdir(parents=False, exist_ok=False)



unprocessed_ct_files = unprocessed_dir.glob('*.ct')
unprocessed_ct_stems = set([f.stem for f in unprocessed_ct_files])
print(f"# unprocessed .ct stems: {len(unprocessed_ct_stems)}")

unprocessed_seq_files = unprocessed_dir.glob('*.seq')
unprocessed_seq_stems = set([f.stem for f in unprocessed_seq_files])
print(f"# unprocessed .seq stems: {len(unprocessed_seq_stems)}")

shared_stems = unprocessed_seq_stems & unprocessed_ct_stems
print(f"# unprocessed shared stems: {len(shared_stems)}")

for stem in tqdm(shared_stems):
    original_seq_path = unprocessed_dir / f"{stem}.seq"
    new_seq_path = processed_dir / f"{stem}.seq"
    shutil.copy(original_seq_path, new_seq_path)

    original_ct_path = unprocessed_dir / f"{stem}.ct"
    new_ct_path = processed_dir / f"{stem}.ct"

    p = subprocess.run([rem_pseudoknots_exe, original_ct_path, new_ct_path], capture_output=True)
    if p.returncode != 0:
        pdb.set_trace()
        raise RuntimeError(f"RemovePseudoknots failed for file: {original_ct_path}")

    new_db_path = processed_dir / f"{stem}.dot"
    p = subprocess.run([ct2dot_exe, new_ct_path, "-1", new_db_path], capture_output=True)
    if p.returncode != 0:
        pdb.set_trace()
        raise RuntimeError(f"ct2dot failed for file: {new_ct_path}")



pdb.set_trace()


print("done")
