from glob import glob
import pandas as pd
import sys
suffix = sys.argv[1]
files = sorted(glob(f"new_variant_data/*_{suffix}.csv"))
print(files)
df = [pd.read_csv(file) for file in files]
df = pd.concat(df, 0)
print(df.shape)

from Bio.PDB.Polypeptide import three_to_one, one_to_three
aa = list("ACDEFGHIKLMNPQRSTVWY")
aa_full_name = ["pr" + one_to_three(a) for a in aa]
probs = df[aa_full_name]
print(probs.head())
info = df[['pdb_id', "pos", "wtAA", "wt_prob"]]
probs = probs.rename(columns=dict(zip(aa_full_name, aa)))
print(probs.head())

df = pd.concat([info, probs], 1)


print(df.shape)
df = df.dropna(0, how="any")
df['gene'] = df['pdb_id'].apply(lambda row: row.split("_")[0])
df = df.iloc[:, 1:]
print(df.shape)
df.to_csv(f"new_variant_data/combined_{suffix}.csv")