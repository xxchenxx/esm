import pandas as pd
data = pd.read_csv("protherm_single.csv")
data = data['PDB_wild']
data = pd.unique(data)
import urllib.request 
print(data)
print(len(data))
for d in data:
    try:
        urllib.request.urlretrieve(f'http://files.rcsb.org/download/{d}.pdb', f'protherm_pdb/{d}.pdb')
    except:
        pass

