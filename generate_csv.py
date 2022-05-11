import pandas as pd
from glob import glob
from Bio.PDB.Polypeptide import three_to_one
from Bio import SeqIO 
import re
# data = glob("filtered_4122_seqs.fasta")
data = glob("PSICOV_seqs.fasta")

seqs = {}
for d in data:
    for record in SeqIO.parse(d, "fasta"):
        name = record.id
        seqs[name] = record.seq

print(len(seqs))
    # print(record.seq)
# assert False
pdb_id = seqs.keys() # data['pdb_id'].unique()
exp_columns = [
    "model",
    "accuracy",
    "pdb_id",
    "chain_id",
    "pos",
    "wtAA",
    "prAA"
]
# print(pdb_id)
for name in pdb_id:
        seq = pd.DataFrame({"sequence": [str(seqs[name]) for _ in range(len(seqs[name]))]})
        seq['model'] = pd.Series([' ' for _ in  range(len(seqs[name]))], index=seq.index)
        seq['accuracy'] = pd.Series([' ' for _ in  range(len(seqs[name]))], index=seq.index)
        seq['pdb_id'] = pd.Series([name for _ in  range(len(seqs[name]))], index=seq.index)
        seq['chain_id'] = pd.Series([' ' for _ in  range(len(seqs[name]))], index=seq.index)
        # print(seq)
        seq['pos'] = list(range(1, len(seqs[name]) + 1))
        seq["wtAA"] = list(seqs[name])
        seq["prAA"] = [0 for _ in range(len(seqs[name]))]

        # seq.to_csv(f"variant_data_0420/{name.replace('/', '_').replace(' ', '_')}.csv")
        # with open(f"variant_data_0420/{name.replace('/', '_').replace(' ', '_')}.txt", 'w') as f:
        seq.to_csv(f"variant_data_0430/{name.replace('/', '_').replace(' ', '_')}.csv")
        with open(f"variant_data_0430/{name.replace('/', '_').replace(' ', '_')}.txt", 'w') as f:
            f.write(str(seqs[name]))
    