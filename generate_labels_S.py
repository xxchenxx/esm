from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold
import pickle
fastas = glob("data/datasets/S_target/*.fasta")
import pandas as pd
temperature_ids = pd.read_csv("temperature_data_taxid.csv")

h = pickle.load(open("hyper_tax_id.pkl", "rb"))
c = pickle.load(open("cryo_tax_id.pkl", "rb"))
m = pickle.load(open("meso_tax_id.pkl", "rb"))
p = pickle.load(open("psychro_tax_id.pkl", "rb"))
t = pickle.load(open("thermo_tax_id.pkl", "rb"))

idxs = {}
fasta_temp = {}
dirty_fasta_temp = {}
for l in tqdm(h):
    idx = int(h[l].split(".")[0])
    idxs[l] = idx
    temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
    temp = temp.replace('C', '')
    temp = temp.replace('c', '')
    if '~' in temp:
        temp = sum(map(float, temp.split("~"))) / 2
    else:
        temp = float(temp)
    if temp >= 75:
        fasta_temp[l] = temp
    else:
        dirty_fasta_temp[l] = temp

for l in tqdm(c):
    idx = int(c[l].split(".")[0])
    idxs[l] = idx
    temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
    temp = temp.replace('C', '')
    temp = temp.replace('c', '')
    if '~' in temp:
        temp = sum(map(float, temp.split("~"))) / 2
    else:
        temp = float(temp)
    if -20 <= temp < 5:
        fasta_temp[l] = temp
    else:
        dirty_fasta_temp[l] = temp

for l in tqdm(m):
    idx = int(m[l].split(".")[0])
    idxs[l] = idx
    temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
    temp = temp.replace('C', '')
    temp = temp.replace('c', '')
    if '~' in temp:
        temp = sum(map(float, temp.split("~"))) / 2
    else:
        temp = float(temp)
    if 25 <= temp < 45:
        fasta_temp[l] = temp
    else:
        dirty_fasta_temp[l] = temp

for l in tqdm(t):
    idx = int(t[l].split(".")[0])
    idxs[l] = idx
    temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
    temp = temp.replace('C', '')
    temp = temp.replace('c', '')
    if '~' in temp:
        temp = sum(map(float, temp.split("~"))) / 2
    else:
        temp = float(temp)
    if 45 <= temp < 75:
        fasta_temp[l] = temp
    else:
        dirty_fasta_temp[l] = temp

for l in tqdm(p):
    idx = int(p[l].split(".")[0])
    idxs[l] = idx
    temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
    temp = temp.replace('C', '')
    temp = temp.replace('c', '')
    if '~' in temp:
        temp = sum(map(float, temp.split("~"))) / 2
    else:
        try:
            temp = float(temp)
        except:
            continue
    if 5 <= temp < 25:
        fasta_temp[l] = temp
    else:
        dirty_fasta_temp[l] = temp

filtered_fasta_temp = {}
for name in tqdm(fastas):
    l = name.split("/")[-1].split(".")[0]
    filtered_fasta_temp[l] = fasta_temp[l]

        
print(len(filtered_fasta_temp))
import pickle
kfold = KFold(n_splits=5, shuffle=True)
name = list(filtered_fasta_temp.keys())
value = list(filtered_fasta_temp.values())


for split_id, (train_id, test_id) in enumerate(kfold.split(name, value)):
    current_split = {"train_names": [name[i] for i in train_id], "train_labels": [value[i] for i in train_id], "test_names": [name[i] for i in test_id], "test_labels": [value[i] for i in test_id]}
    pickle.dump(current_split, open(f"S_target_new_{split_id}.pkl", "wb"))

pickle.dump(filtered_fasta_temp, open("S_noisy.pkl", "wb"))