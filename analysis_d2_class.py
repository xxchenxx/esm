import pickle
from tqdm import tqdm
a = pickle.load(open("d2_0.pkl", "rb"))
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
    
    if not flag:
        for l in psychro:
            # print(l)
            if idname in l:
                mapping[name] = 'psychro'
                flag = True
                break
    
    if not flag:
        for l in thermo:
            # print(l)
            if idname in l:
                mapping[name] = 'thermo'
                flag = True
                break

    if not flag:
        for l in meso:
            # print(l)
            if idname in l:
                mapping[name] = 'meso'
                flag = True
                break

from glob import glob
meso = glob("ThermoMLL-master/data/mesophilic/*")
meso = {name:open(name).read() for name in meso}
hyper = glob("ThermoMLL-master/data/hyperphilic/*")
hyper = {name:open(name).read() for name in hyper}
thermo = glob("ThermoMLL-master/data/thermophilic/*")
thermo = {name:open(name).read() for name in thermo}
cryo = glob("ThermoMLL-master/data/cryophilic/*")
cryo = {name:open(name).read() for name in cryo}
psychro = glob("ThermoMLL-master/data/psychrophilic/*")
psychro = {name:open(name).read() for name in psychro}

import pickle
# mapping = pickle.load(open("d2_mapping.pkl","rb"))
for name in mapping:
    if mapping[name] is None:
        for key in meso:
            if name in meso[key]:
                mapping[name] = 'meso'
                break
        for key in hyper:
            if name in hyper[key]:
                mapping[name] = 'hyper'
                break
        for key in thermo:
            if name in thermo[key]:
                mapping[name] = 'thermo'
                break
        for key in cryo:
            if name in cryo[key]:
                mapping[name] = 'cryo'
                break
        for key in psychro:
            if name in psychro[key]:
                mapping[name] = 'psychro'
                break
pickle.dump(mapping, open("d2_mapping_class.pkl", 'wb'))
