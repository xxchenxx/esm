#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import random
from sched import scheduler
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer

from scipy.stats import spearmanr, pearsonr
from esm.utils import PGD_regression, PGD_regression_amino
import wandb
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
    parser.add_argument("--mix", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--aadv", action="store_true")
    parser.add_argument("--wandb-name", type=str, default='protein')
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1e-3)
    return parser

def pruning_model(model, px):
    

    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        #if 'self_attn' in name and (not 'k' in name) and isinstance(m, nn.Linear):
        #    print(f"Pruning {name}")
        #    parameters_to_prune.append((m,'weight'))
        if isinstance(m, TransformerLayer):
            print(f"Pruning {name}.fc1")
            parameters_to_prune.append((m.fc1,'weight'))
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
    wandb.init(project=f"protein", entity="xxchen", name=args.wandb_name)
    wandb.config.update(vars(args))
    best = 0
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location, num_classes=args.num_classes, noise_aug=args.noise)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    import sys

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, shuffle=True,
    )
    #print(f"Read {args.fasta_file} with {len(train_sets[0])} sequences")

    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, #batch_sampler=test_data_loader
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    if args.pruning_ratio > 0:
        pruning_model(model, args.pruning_ratio)

    model = model.cuda().eval()
    linear = nn.Sequential( nn.Linear(1280, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 5)).cuda()
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=5e-2)
    linear.load_state_dict(torch.load("linear-supervised-finetuned-S_classification.pt", map_location="cpu"))
    step = 0
    for epoch in range(1):
        with torch.no_grad():
            outputs = []
            tars = []
            for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(test_data_loader)} batches ({toks.size(0)} sequences)"
                )
                if torch.cuda.is_available() and not args.nogpu:
                    toks = toks.to(device="cuda", non_blocking=True)
                # The model is trained on truncated sequences and passing longer ones in at
                # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
                if args.truncate:
                    toks = toks[:, :1022]
                out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)
                hidden = out['hidden']
                logits = linear(hidden)
                labels = torch.tensor(labels).cuda().long()
            

                outputs.append(torch.topk(logits.reshape(-1, args.num_classes), 1)[1].view(-1))
                tars.append(labels.reshape(-1))
            
            outputs = torch.cat(outputs, 0)
            tars = torch.cat(tars, 0)
            print("EVALUATION:", float((outputs == tars).float().sum() / tars.nelement()))
            acc = (outputs == tars).float().sum() / tars.nelement()
            precision = ((outputs == tars).float() * (outputs == 1).float()).sum() / (outputs == 1).float().sum()
            print("PRECISION:", precision)
            wandb.log({"accuracy": acc}, step=step)
            wandb.log({"precision": precision}, step=step)
    print(best)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)