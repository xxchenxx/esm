import pickle
from tqdm import tqdm

a = pickle.load(open("d1_0.pkl", "rb"))
cryo = pickle.load(open("cryo_tax_id.pkl", "rb"))
hyper = pickle.load(open("hyper_tax_id.pkl", "rb"))
meso = pickle.load(open("meso_tax_id.pkl", "rb"))
thermo = pickle.load(open("thermo_tax_id.pkl", "rb"))
psychro = pickle.load(open("psychro_tax_id.pkl", "rb"))

names = a['train_names'] + a['test_names']
mapping = {name: None for name in names}
for name in tqdm(names):
    idname = name.split("/")[-1].split(".")[0]
    # print(idname)
    flag = False
    for l in hyper:
        # print(l)
        if idname in l:
            mapping[name] = 'hyper'
            flag = True
            break
    if not flag:
        for l in cryo:
            # print(l)
            if idname in l:
                mapping[name] = 'cryo'
                flag = True
                break
    
pickle.dump(mapping, open("d1_mapping_class.pkl", 'wb'))
