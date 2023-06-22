#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib
import torch
import torch.nn as nn
from esm import Alphabet, CSVDNABatchedDataset, ProteinBertModel, pretrained, MSATransformer
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertConfig, DNATokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from config import create_parser

from transformers import glue_output_modes as output_modes
import logging
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

def combine_tokens(toks):
    toks = toks.split(" ")

    string = toks[0]

    end_toks = string[-2:]

    for i in range(1, len(toks)):
        if not toks[i].startswith(end_toks):
            return None
        string = f"{string}{toks[i][-1]}"
        end_toks = toks[i][-2:]
    return string

def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    
    tokenizer_class = DNATokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    pad_on_left = bool(args.model_type in ["xlnet"])
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    pad_token_segment_id = 4 if args.model_type in ["xlnet"] else 0
    output_mode = output_modes['dnaprom']

    dataset = CSVDNABatchedDataset.from_file(args.csv_file, args, tokenizer, output_mode, pad_on_left, pad_token, pad_token_segment_id)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length, with_attention=True), batch_size=1
    )
    print(f"Read {args.csv_file} with {len(dataset)} sequences")

    args.output_dir = pathlib.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    ## DNA model begins

    config_class = BertConfig
    model_class = BertGenerationDecoder
    

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=2,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    if args.model_type in ["dnalong", "dnalongcat"]:
        assert args.max_seq_length % 512 == 0
    config.split = int(args.max_seq_length / 512)
    config.rnn = args.rnn
    config.num_rnn_layer = args.num_rnn_layer
    config.rnn_dropout = args.rnn_dropout
    config.rnn_hidden = args.rnn_hidden
    config.is_decoder = True
    config.add_cross_attention=True
    config.bos_token_id = 2
    config.eos_token_id = 3

    decoder = model_class(config)
    decoder.cuda()
    projector = nn.Linear(1280, 768, bias=False).cuda()
    decoder.eval()
    model.eval()
    projector.eval()


    decoder.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    projector.load_state_dict(torch.load(args.checkpoint)['projector'])
    num_return_sequences = 3
    df = pd.DataFrame({'label': []})
    for i in range(num_return_sequences):
        df[f'generated_{i}'] = []
    for batch_idx, (labels, strs, toks, _, _) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            with torch.no_grad():
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            # print(out["representations"][repr_layers[-1]].shape)
            labels_mask_length = (labels.sum(0) > 0).sum()
            labels = labels[:, torch.arange(labels_mask_length + 2)]
            encoder_hidden_states = projector(out["representations"][repr_layers[-1]])
            try:
                outputs = decoder.generate(encoder_hidden_states=encoder_hidden_states, num_beams=3, num_return_sequences=num_return_sequences, top_k=5, max_length=50, repetition_penalty=1, top_p=0.9, temperature=1.5)

                rows = [combine_tokens(tokenizer.decode(labels[0], skip_special_tokens=True))]
                for i in range(num_return_sequences):
                    output = tokenizer.decode(outputs[0][i], skip_special_tokens=True)
                    combined_tokens = combine_tokens(output)
                    rows.append(combined_tokens)
            except:
                rows = [None]
                for i in range(num_return_sequences):
                    rows.append(None)
            df.loc[batch_idx] = rows
            df.to_csv(f"{args.output_name}.csv")
            # print('Label: {}'.format(tokenizer.decode(labels[0], skip_special_tokens=True)))    
            # print('\n')

    

def main():
    parser = create_parser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument("--output-name", type=str)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
