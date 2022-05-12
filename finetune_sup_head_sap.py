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
import wandb
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds, PickleBatchedDataset, FireprotDBBatchedDataset
from esm.modules import TransformerLayer
from esm.utils import PGD_classification, PGD_classification_amino


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
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--aadv", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wandb-name", type=str, default="protein")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1e-3)
    return parser


def set_seed(args):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main(args):

    set_seed(args)
    best = 0
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location, num_classes=args.num_classes, noise_aug=args.noise, rank=args.rank)
    model.load_state_dict(torch.load("checkpoint_sequencemoco.pt", map_location='cpu')['state_dict'])
    model.eval()
    wandb.init(project=f"protein", entity="xxchen", name=args.wandb_name)
    wandb.config.update(vars(args))
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    import sys

    train_set = PickleBatchedDataset.from_file(args.split_file, True, args.fasta_file)
    test_set = PickleBatchedDataset.from_file(args.split_file, False, args.fasta_file)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, collate_fn=alphabet.get_batch_converter(), batch_size=args.batch_size, shuffle=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_set, collate_fn=alphabet.get_batch_converter(), batch_size=4, #batch_sampler=test_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    model = model.cuda().eval()
    linear = nn.Sequential( nn.Linear(1280, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, args.num_classes)).cuda()
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=5e-2)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=1, epochs=int(20))
    step = 0
    for epoch in range(6):
        model.eval()
        for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
            step += 1
            with torch.autograd.set_detect_anomaly(True):
                print(
                    f"Processing {batch_idx + 1} of {len(train_data_loader)} batches ({toks.size(0)} sequences)"
                )
                toks = toks.cuda()
                if args.truncate:
                    toks = toks[:, :1022]
                with torch.no_grad():
                    out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)

                hidden = out['hidden']

                labels = torch.tensor(labels).cuda().long()
                if args.mix:
                    lam = np.random.beta(0.2, 0.2)
                    rand_index = torch.randperm(hidden.size()[0]).cuda()
                    labels_all_a = labels
                    labels_all_b = labels[rand_index]
                    hiddens_a = hidden
                    hiddens_b = hidden[rand_index]
                    hiddens = lam * hiddens_a + (1 - lam) * hiddens_b
                    hiddens = linear(hiddens)
                    loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_a) * lam + \
                        F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels_all_b) * (1 - lam)
                elif args.adv:
                    hidden_adv = PGD_classification(hidden, linear, labels, steps=args.steps, eps=3/255, num_classes=args.num_classes, gamma=args.gamma)
                    hiddens_adv = linear(hidden_adv)
                    hiddens_clean = linear(hidden)
                    loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
                elif args.aadv:
                    hidden_adv = PGD_classification_amino(hidden, linear, labels, steps=args.steps, eps=3/255, num_classes=args.num_classes, gamma=args.gamma)
                    hiddens_adv = linear(hidden_adv)
                    hiddens_clean = linear(hidden)
                    loss = (F.cross_entropy(hiddens_adv.view(hiddens_adv.shape[0], args.num_classes), labels) + F.cross_entropy(hiddens_clean.view(hiddens_clean.shape[0], args.num_classes), labels)) / 2
                else:
                    hiddens = linear(hidden)
                    loss = F.cross_entropy(hiddens.view(hiddens.shape[0], args.num_classes), labels)
                loss.backward()
                optimizer.step()
                linear.zero_grad()
                print(loss.item())
                if (step + 1) % 20000 == 0:
                    with torch.no_grad():
                        outputs = []
                        tars = []
                        for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
                            print(
                                f"Processing {batch_idx + 1} of {len(test_data_loader)} batches ({toks.size(0)} sequences)"
                            )
                            if torch.cuda.is_available() and not args.nogpu:
                                toks = toks.to(device="cuda", non_blocking=True)
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
                        if acc > best:
                            torch.save(linear.state_dict(), f"head-classification-{args.idx}.pt")
                            best = acc
        lr_scheduler.step()
        model.eval()
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
                
                print(loss.item())

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
            if acc > best:
                torch.save(linear.state_dict(), f"head-classification-{args.idx}.pt")
                best = acc
    print(best)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
