import pandas as pd
from glob import glob
from tqdm import tqdm
df = None
dfs = []
for name in tqdm(glob("variant_data_0415/*_1_esm_translated.csv")):
    df = pd.read_csv(name)
    df['pdb_id'] = name.split("/")[1].split("_")[0]
    df['chain_id'] = 'A'
    dfs.append(df)
    
dfs = pd.concat(dfs, 0)
# del dfs['acc']
dfs.to_csv("esm.csv", index=False)