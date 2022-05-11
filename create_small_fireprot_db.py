import pandas as pd
from glob import glob


data = pd.read_csv("combined_features.csv")
pdb_files = glob("pdb/*")

pdb_ids = []
sequences = []
targets = []

for pdb in pdb_files:
    pdb_id = pdb.split("/")[1].split(".")[0]
    row = data[data['pdb_id'] == pdb_id].iloc[0]
    sequences.append(row['sequence'])
    pdb_ids.append(pdb_id)
    if pd.isna(row['dTm']): targets.append(float(row['tm']))
    else: targets.append(float(row['tm']) + float(row['dTm']))

df = pd.DataFrame({'ids': pdb_ids, 'sequences': sequences, 'targets': targets})
df.to_csv('fireprot.csv')