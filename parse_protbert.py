import torch
import pandas as pd
import numpy as np
import sys
from glob import glob
names = glob("variant_data_0430/*.txt")

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
from tqdm import tqdm
for name in tqdm(names):
   
    data = pd.read_csv(name)
    name = name.split(".")[0]
    data = data.iloc[:, :-5]
    metrics = torch.load(f"{name}_prot.pth.tar", map_location='cpu')

    sequence = open(name + ".txt").read()
    
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    ids = tokenizer.batch_encode_plus([' '.join(aa)], add_special_tokens=False, pad_to_max_length=False)
    mapping = ids['input_ids'][0]
    # print(metrics.sum(1).shape)
    metrics = metrics[:, mapping]
    metrics = metrics / metrics.sum(1).view(-1, 1).repeat(1, metrics.shape[1])
    metrics = np.array(metrics)
    from Bio.PDB.Polypeptide import three_to_one, one_to_three
    aa_full_name = ["pr" + a for a in aa]

    prob = pd.DataFrame(metrics, columns=aa_full_name)

    prob = pd.concat([data, prob], 1)
    try:
        prob['wtAA'] = pd.Series([a for a in list(sequence)])
        prob.to_csv(f"{name}_protbert.csv")
    except:
        pass