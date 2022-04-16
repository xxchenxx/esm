import torch
import pandas as pd
import numpy as np
import sys

data = pd.read_csv(f"{sys.argv[1]}.csv")

data = data.iloc[:, :-5]

metrics = torch.load(f"{sys.argv[1]}_prot.pth.tar", map_location='cpu')
sequence = sys.argv[2]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
aa = list("ACDEFGHIKLMNPQRSTVWY")
ids = tokenizer.batch_encode_plus([' '.join(aa)], add_special_tokens=False, pad_to_max_length=False)
mapping = ids['input_ids'][0]
print(metrics.sum(1).shape)
metrics = metrics[:, mapping]
metrics = metrics / metrics.sum(1).view(-1, 1).repeat(1, metrics.shape[1])
metrics = np.array(metrics)
from Bio.PDB.Polypeptide import three_to_one, one_to_three
aa_full_name = ["pr" + one_to_three(a) for a in aa]

prob = pd.DataFrame(metrics, columns=aa_full_name)

prob = pd.concat([data, prob], 1)
prob['wtAA'] = pd.Series([one_to_three(a) for a in list(sequence)])
prob.to_csv(f"{sys.argv[1]}_protbert.csv")