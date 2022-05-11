import pickle
from tqdm import tqdm

a = pickle.load(open("S_target_0.pkl", "rb"))
cryo = pickle.load(open("cryo_tax_id.pkl", "rb"))
hyper = pickle.load(open("hyper_tax_id.pkl", "rb"))
meso = pickle.load(open("meso_tax_id.pkl", "rb"))
thermo = pickle.load(open("thermo_tax_id.pkl", "rb"))
psychro = pickle.load(open("psychro_tax_id.pkl", "rb"))

names = a['train_names'] + a['test_names']
mapping = {name: None for name in names}
mapping_class = {name: None for name in names}
"""
for name in tqdm(names):
    idname = name.split("/")[-1].split(".")[0]
    # print(idname)
    flag = False
    for l in hyper:
        # print(l)
        if idname in l:
            mapping[name] = int(hyper[l].split(".")[0])
            mapping_class[name] = 'hyper'
            flag = True
            break
    if not flag:
        for l in cryo:
            # print(l)
            if idname in l:
                mapping[name] = int(cryo[l].split(".")[0])
                mapping_class[name] = 'cryo'
                flag = True
                break
    if not flag:
        for l in meso:
            # print(l)
            if idname in l:
                mapping[name] = int(meso[l].split(".")[0])
                mapping_class[name] = 'meso'
                flag = True
                break
    if not flag:
        for l in thermo:
            # print(l)
            if idname in l:
                mapping[name] = int(thermo[l].split(".")[0])
                mapping_class[name] = 'thermo'
                flag = True
                break
    if not flag:
        for l in psychro:
            # print(l)
            if idname in l:
                mapping[name] = int(psychro[l].split(".")[0])
                mapping_class[name] = 'psychro'
                flag = True
                break
"""
for l in hyper:
    mapping[l] = int(hyper[l].split(".")[0])
    mapping_class[l] = 'hyper'
for l in cryo:
    mapping[l] = int(cryo[l].split(".")[0])
    mapping_class[l] = 'cryo'
for l in meso:
    mapping[l] = int(meso[l].split(".")[0])
    mapping_class[l] = 'meso'
for l in psychro:
    mapping[l] = int(psychro[l].split(".")[0])
    mapping_class[l] = 'psychro'
for l in thermo:
    mapping[l] = int(thermo[l].split(".")[0])
    mapping_class[l] = 'thermo'
final_mapping = {}
final_mapping_class = {}
for name in names:
    final_mapping[name] = mapping[name]
    final_mapping_class[name] = mapping_class[name]
print(len(final_mapping))
print(len(names))
pickle.dump(final_mapping, open("S_mapping.pkl", 'wb'))
pickle.dump(final_mapping_class, open("S_mapping_class.pkl", 'wb'))
