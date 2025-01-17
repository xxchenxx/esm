#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, CSVBatchedDataset, creating_ten_folds


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

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = CSVBatchedDataset.from_file("combined_features.csv")
    train_sets, test_sets = creating_ten_folds(dataset)
    train_batches = train_sets[9].get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    train_data_loader = torch.utils.data.DataLoader(
        train_sets[9], collate_fn=alphabet.get_batch_converter(), batch_sampler=train_batches
    )
    #print(f"Read {args.fasta_file} with {len(train_sets[0])} sequences")

    test_batches = test_sets[0].get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)

    test_data_loader = torch.utils.data.DataLoader(
        test_sets[0], collate_fn=alphabet.get_batch_converter(), batch_sampler=test_batches
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for epoch in range(1):
        model.train()
        for batch_idx, (labels, strs, targets, toks) in enumerate(train_data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(train_batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            #print(toks)
            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts, return_temp=True)

            temp = out['temp'] * 10
            targets = torch.tensor(targets).cuda().float()
            loss = (torch.nn.functional.mse_loss(temp[:,0].view(-1), targets.view(-1)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(loss.item())
        model.eval()
        with torch.no_grad():
            outputs = []
            tars = []
            for batch_idx, (labels, strs, targets, toks) in enumerate(test_data_loader):
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
                temp = out['temp'] * 10
                targets = torch.tensor(targets).cuda().float()
                outputs.append(temp[:,0].view(-1).cpu().numpy())
                tars.append(targets.view(-1).cpu().numpy())
            print(outputs)
            print(tars)
            import numpy as np
            outputs = np.concatenate(outputs, 0)
            tars = np.concatenate(tars, 0)
            print(np.corrcoef(outputs, tars))
    torch.save(model.state_dict(), "supervised-finetuned.pt")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
