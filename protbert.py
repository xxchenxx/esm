import torch
from transformers import  BertForMaskedLM, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm

tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
model =  BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args()
from glob import glob
files = glob("variant_data_0430/*.txt")
for name in files:
    args.sequence = open(name).read()
    ids = tokenizer.batch_encode_plus([' '.join(list(args.sequence))], add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    print(len(input_ids))
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
    print(embedding.shape)

    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        print(seq_len)
        seq_emd = embedding[seq_num][1:seq_len-1]

    seq_emd = torch.softmax(seq_emd, -1)
    torch.save(seq_emd, f"{name.split('.')[0] + '_prot.pth.tar'}")
