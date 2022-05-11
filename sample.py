from glob import glob

from importlib_metadata import files

files1 = glob("meso*/*")
files2 = glob("cryophilic/*")

import random
files1_sampled = random.sample(files1, 4)
files2_sampled = random.sample(files2, 4)

all_seqs = []

for file in files1_sampled:
    all_seqs.append(open(file).read())

for file in files2_sampled:
    all_seqs.append(open(file).read())

with open("sampled.fasta", "w") as f:
    f.write('\n'.join(all_seqs))