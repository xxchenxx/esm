import numpy as np
import esm
import pickle
import torch
import torch.nn as nn
import sys
split = sys.argv[1]
split = pickle.load(open(f"d1_{split}_classification.pkl", "rb"))
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.cuda()
linear = nn.Sequential( nn.Linear(512, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 2)).cuda() 
optimizer = torch.optim.AdamW(linear.parameters(), lr=2e-2, weight_decay=5e-2)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, steps_per_epoch=1, epochs=int(20))

for epoch in range(4):
    outputs = []
    labels = []
    for name, label in zip(split['train_names'], split['train_labels']):
        fpath = f"d1/d1_clean/{name}/unrelaxed_model_1_ptm.pdb"
        # print(fpath)
        structure = esm.inverse_folding.util.load_structure(fpath, 'A')
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        coords = torch.from_numpy(coords).cuda()
        rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        # print(rep.shape)
        output = linear(rep.mean(0, keepdim=True))
        outputs.append(output)
        labels.append(torch.tensor(label).long().cuda())

        if len(outputs) == 4:
            outputs = torch.cat(outputs, 0)
            labels = torch.stack(labels, 0)
            # print(outputs.shape)
            # print(labels.shape)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = []
            labels = []
    if len(outputs) != 0:
        outputs = torch.cat(outputs, 0)
        labels = torch.stack(labels, 0)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        outputs = []
        labels = []
    lr_scheduler.step()
    for name, label in zip(split['train_names'], split['train_labels']):
        fpath = f"d1/d1_clean/{name}/unrelaxed_model_1_ptm.pdb"
        # print(fpath)
        structure = esm.inverse_folding.util.load_structure(fpath, 'A')
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)

        output = linear(rep.unsqueeze(-1))
        outputs.append(output)
        labels.append(label.long())
    
    outputs = torch.cat(outputs, 0)
    labels = torch.stack(labels, 0)
    acc = (outputs == labels).float().sum() / labels.nelement()
    precision = ((outputs == labels).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum() 
    print(acc)
    print(precision)