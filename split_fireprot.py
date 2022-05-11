import pandas as pd
a = pd.read_csv("fireprot.csv")
names = list(a['ids'])
temperature = list(a['targets'])
seqs = list(a['sequences'])
print(names)
print(temperature)
print(seqs)
names = list(map(lambda x: x , names))
for name, t, s in zip(names, temperature, seqs):
    with open(f"fireprot/{name}.fasta", "w") as f:
        f.write(">" + name)
        f.write("\n")
        f.write(s)

from sklearn.model_selection import KFold
print(len(names))
import pickle
kfold = KFold(n_splits=10, shuffle=True)

for split_id, (train_id, test_id) in enumerate(kfold.split(names, temperature)):
    current_split = {"train_names": [names[i] for i in train_id], "train_labels": [temperature[i] for i in train_id], "test_names": [names[i] for i in test_id], "test_labels": [temperature[i] for i in test_id]}
    pickle.dump(current_split, open(f"fireprot_regression_{split_id}.pkl", "wb"))