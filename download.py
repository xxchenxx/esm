import pandas as pd
data = pd.read_csv("combined_features.csv")
data = data['pdb_id']
data = pd.unique(data)
import urllib.request 
print(data)
print(len(data))
for d in data:
    try:
        urllib.request.urlretrieve(f'http://files.rcsb.org/download/{d}.pdb', f'pdb/{d}.pdb')
    except:
        pass