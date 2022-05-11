from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold
import pickle
labels = glob("data/datasets/S_target/*.fasta")
import pandas as pd
temperature_ids = pd.read_csv("temperature_data_taxid.csv")

h = pickle.load(open("cryo_tax_id.pkl", "rb"))
# c = pickle.load(open("cryo_tax_id.pkl", "rb"))

idxs = []
d2_temp = {}
for name in tqdm(labels):
    idname = name.split("/")[-1].split(".")[0]
    # print(idname)
    flag = False
    for l in h:
        # print(l)
        if idname in l:
            idx = int(h[l].split(".")[0])
            idxs.append(idx)
print(len(set(idxs)))
'''
            temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
            temp = temp.replace('C', '')
            temp = temp.replace('c', '')
            if '~' in temp:
                temp = sum(map(float, temp.split("~"))) / 2
            else:
                temp = float(temp)
            d2_temp[idname] = temp
            flag = True
            break
    if not flag:
        for l in c:
            if idname in l:
                idx = int(c[l].split(".")[0])
                temp = (temperature_ids[temperature_ids['taxid'] == idx]['optimum_temperature'].iloc[0])
                temp = temp.replace('C', '')
                temp = temp.replace('c', '')
                if '~' in temp:
                    temp = sum(map(float, temp.split("~"))) / 2
                else:
                    temp = float(temp)
                d2_temp[idname] = temp
                flag = True
                break
            
print(len(d2_temp))
import pickle
kfold = KFold(n_splits=10, shuffle=True)
name = list(d2_temp.keys())
value = list(d2_temp.values())


for split_id, (train_id, test_id) in enumerate(kfold.split(name, value)):
    current_split = {"train_names": [name[i] for i in train_id], "train_labels": [value[i] for i in train_id], "test_names": [name[i] for i in test_id], "test_labels": [value[i] for i in test_id]}
    pickle.dump(current_split, open(f"d1_{split_id}.pkl", "wb"))
'''