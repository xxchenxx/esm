import esm
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.cuda()
from glob import glob
import torch
import torch.nn as nn
torch.cuda.is_available()
import sys
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
split = pickle.load(open(sys.argv[1], 'rb'))
train_names = split['train_names']
test_names = split['test_names']

train_labels = split['train_labels']
test_labels = split['test_labels']
train_structures = [esm.inverse_folding.util.load_structure(f'{sys.argv[2]}/{fpath}/unrelaxed_model_1_ptm.pdb', 'A') for fpath in train_names]
test_structures = [esm.inverse_folding.util.load_structure(f'{sys.argv[2]}/{fpath}/unrelaxed_model_1_ptm.pdb', 'A') for fpath in test_names]
train_structures = [esm.inverse_folding.util.extract_coords_from_structure(structure)[0] for structure in train_structures]
test_structures = [esm.inverse_folding.util.extract_coords_from_structure(structure)[0] for structure in test_structures]

linear = nn.Sequential( nn.Linear(512, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 1)).cuda()
optimizer = torch.optim.AdamW(linear.parameters(), lr=2e-2, weight_decay=5e-2)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-2, steps_per_epoch=1, epochs=int(20))
for epoch in range(4):
    randperm = np.random.permutation(len(train_names))
    train_structures = [train_structures[i] for i in randperm]
    train_labels = [train_labels[i] for i in randperm]
    coords = []
    for i in range(len(train_names) // 4):
        batch_structures = train_structures[i*4:(i+1)*4]
        labels = torch.tensor(train_labels[i*4:(i+1)*4]).float().cuda()
        reps = []
        for structure in batch_structures:
            reps.append(esm.inverse_folding.util.get_encoder_output(model, alphabet, torch.tensor(structure).cuda()).mean(0))
        reps = torch.stack(reps, 0)
        hiddens = linear(reps)
        loss = torch.nn.functional.mse_loss(hiddens.view(hiddens.shape[0], 1), labels)
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    outputs = []
    tars = []
    with torch.no_grad():
        for i in range(len(test_names) // 4):
            batch_structures = test_structures[i*4:(i+1)*4]
            labels = torch.tensor(test_labels[i*4:(i+1)*4]).float().cuda()
            reps = []
            for structure in batch_structures:
                reps.append(esm.inverse_folding.util.get_encoder_output(model, alphabet, torch.tensor(structure).cuda()).mean(0))
            reps = torch.stack(reps, 0)
            hiddens = linear(reps)
            outputs.append(hiddens.reshape(-1, 1).view(-1) * 10)
            tars.append(labels.reshape(-1))
            
        outputs = torch.cat(outputs, 0).detach().cpu().numpy()
        tars = torch.cat(tars, 0).detach().cpu().numpy()
        spearman = spearmanr(outputs, tars)[0]
        print("EVALUATION:", spearman)
        pearson = pearsonr(outputs, tars)[0]
        print("EVALUATION:", pearson)