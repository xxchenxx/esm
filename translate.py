import pandas as pd
import sys
from glob import glob
from Bio.PDB.Polypeptide import three_to_one
csv = pd.read_csv("psicov_all_ss_cnn-final.csv")
col = csv.columns
from tqdm import tqdm
files = glob("variant_data_0430/*_protbert.csv")
dfs = []
for file in tqdm(files):
    to_translate = pd.read_csv(file)

    df = pd.DataFrame()
    for c in col:
        if c in to_translate.columns:
            df[c] = to_translate[c]
        else:
            # print(c)
            if c.startswith('pr'):
                try:
                    aa = c[2:]
                    c = 'pr' + three_to_one(aa)
                    df[c] = to_translate[c]
                except:
                    df[c] = None
            else:
                df[c] = None
    for i in range(1, df.shape[0] + 1):
        df['pos'].iloc[i - 1] = i
        #print(df['wtAA'].iloc[i - 1])
        try:
            df['wt_prob'].iloc[i - 1] = df['pr' + df['wtAA'].iloc[i - 1]].iloc[i - 1] 
        except:
            break

    df['model'] = 'esm' 
    df['pdb_id'] = file.split("/")[1][:4]
    df['chain_id'] = file.split("/")[1].split(".")[0][5:-4]
    # print(file.split("/")[1].split(".")[0][5:-9])
    dfs.append(df)

dfs = pd.concat(dfs, 0)
dfs.to_csv("protbert.csv")