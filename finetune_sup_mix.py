#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer

import torch.nn.functional as F

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )

    parser.add_argument(
        "--split_file",
        type=str,
        help="fold",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        help="num_classes",
        default=2, 
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="learning rates",
        default=1e-6, 
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument("--idx", type=str, default='0')
    parser.add_argument("--pruning_ratio", type=float, default=0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mixup", action="store_true")
    return parser

def pruning_model(model, px):
    

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if 'self_attn' in name and isinstance(m, nn.Linear):
            print(f"Pruning {name}")
            parameters_to_prune.append((m,'weight'))
        elif isinstance(m, TransformerLayer):
            print(f"Pruning {name}.fc2")
            parameters_to_prune.append((m.fc2,'weight'))
            print(f"Pruning {name}.fc2")
            parameters_to_prune.append((m.fc2,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def set_seed(args):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):

    set_seed(args)
    best = 0
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location, num_classes=args.num_classes)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    import sys

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    train_batches = train_set.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_sampler=train_batches
    )
    #print(f"Read {args.fasta_file} with {len(train_sets[0])} sequences")

    test_batches = test_set.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), batch_sampler=test_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.pruning_ratio > 0:
        pruning_model(model, args.pruning_ratio)

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    for epoch in range(30):
        model.train()
        hiddens = []
        labels_all = []
        for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
            with torch.autograd.set_detect_anomaly(True):
                print(
                    f"Processing {batch_idx + 1} of {len(train_batches)} batches ({toks.size(0)} sequences)"
                )
                toks = toks.cuda()
                if args.truncate:
                    toks = toks[:, :1022]
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)

                #logits = out['cls_logits']
                labels = torch.tensor(labels).cuda().long()
                hidden = out['hidden']
                hiddens.append(hidden)
                labels_all.append(labels.view(-1))
                if len(hiddens) > 1 or hiddens[0].shape[0] > 1:
                    hiddens = [hidden[:, 0] for hidden in hiddens]
                    hiddens = torch.cat(hiddens, 0).squeeze(1)
                    labels_all = torch.cat(labels_all, 0)
                    labels_one_hot = torch.zeros((labels_all.shape[0], 2)).cuda()
                    for i in range(labels_all.shape[0]):
                        labels_one_hot[i, labels_all[i]] = 1
                    lam = np.random.beta(0.2, 0.2)
                    rand_index = torch.randperm(hiddens.size()[0]).cuda()
                    labels_all_a = labels_one_hot
                    labels_all_b = labels_one_hot[rand_index]

                    hiddens_a = hiddens
                    hiddens_b = hiddens[rand_index]

                    labels_all = lam * labels_all_a + (1 - lam) * labels_all_b
                    hiddens = lam * hiddens_a + (1 - lam) * hiddens_b

                    hiddens = model.classification_head(hiddens)
                    
                    loss = -(F.log_softmax(hiddens, 1) * labels_all).sum(1).mean(0)
                    
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    print(loss.item())
                    hiddens = []
                    labels_all = []
        scheduler.step()
        model.eval()
        with torch.no_grad():
            outputs = []
            tars = []
            
            for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(test_batches)} batches ({toks.size(0)} sequences)"
                )
                if torch.cuda.is_available() and not args.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)
                # The model is trained on truncated sequences and passing longer ones in at
                # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
                if args.truncate:
                    toks = toks[:, :1022]
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)
                
                logits = out['cls_logits']
                labels = torch.tensor(labels).cuda().long()
                outputs.append(torch.topk(logits[:,0].reshape(-1, args.num_classes), 1)[1].view(-1))
                tars.append(labels.reshape(-1))
            
            outputs = torch.cat(outputs, 0)
            tars = torch.cat(tars, 0)
            print("EVALUATION:", float((outputs == tars).float().sum() / tars.nelement()))
            acc = (outputs == tars).float().sum() / tars.nelement()
            if acc > best:
                torch.save(model.state_dict(), f"supervised-finetuned-{args.idx}.pt")
                best = acc
    print(best)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
