import torch
import pandas as pd
import numpy as np
import sys

data = pd.read_csv(f"{sys.argv[1]}.csv")

data = data.iloc[:, :-5]

metrics = np.load(f"{sys.argv[1]}.npy", allow_pickle=True)
sequence = sys.argv[2]

from Bio.PDB.Polypeptide import three_to_one, one_to_three

aa = list("ACDEFGHIKLMNPQRSTVWY")
aa_full_name = ["pr" + one_to_three(a) for a in aa]

metrics = np.array(metrics)[0]
metrics = np.exp(metrics)


prob = pd.DataFrame(metrics, columns=aa_full_name)

prob = pd.concat([data, prob], 1)
prob['wtAA'] = pd.Series([one_to_three(a) for a in list(sequence)])
prob.to_csv(f"{sys.argv[1]}_esm.csv")