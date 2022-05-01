import torch
import pandas as pd
import numpy as np
import sys
from glob import glob
for name in glob("variant_data_0430/*.csv"):
    if 'protbert' in name or 'esm' in name: continue
    data = pd.read_csv(name)

    data = data.iloc[:, :-5]
    name = name.split(".")[0]
    try:
        metrics = np.load(f"{name}.npy", allow_pickle=True)
    except:
        continue
    print(name)
    print(type(metrics))
    sequence = open(name + ".txt").read()

    from Bio.PDB.Polypeptide import three_to_one, one_to_three

    aa = list("ACDEFGHIKLMNPQRSTVWY")
    aa_full_name = ["pr" + a for a in aa]
    print(metrics.shape)
    metrics = np.array(metrics)
    metrics = np.exp(metrics)
    

    prob = pd.DataFrame(metrics, columns=aa_full_name)

    prob = pd.concat([data, prob], 1)
    prob['wtAA'] = pd.Series([a for a in list(sequence)])
    prob.to_csv(f"{name}_esm.csv")