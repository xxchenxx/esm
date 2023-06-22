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
import matplotlib.pyplot as plt
from esm import Alphabet, CSVDNABatchedDataset, ProteinBertModel, pretrained, MSATransformer
from transformers import BertGenerationEncoder, BertGenerationDecoder, BertConfig, DNATokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from config import create_parser

from transformers import glue_output_modes as output_modes
import logging
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()

    if args.no_pretrained_encoder:
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        model.apply(init)
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

    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length, with_attention=True), shuffle=True, batch_size=16
    )
    print(f"Read {args.csv_file} with {len(dataset)} sequences")

    args.output_dir = pathlib.Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

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
    print(config)
    
    # Encoder: takes protein seq as input -> output: protein embedding BS x SEQ x HIDDEN DIM
    # Decoder: takes protein embedding as input -> output: DNA nucleotide
    decoder = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # decoder = model_class(config)
    decoder.cuda()
    projector = nn.Linear(1280, 768, bias=False).cuda()
    nn.init.kaiming_uniform_(projector.weight)
    optimizer_projector = torch.optim.SGD(projector.parameters(), lr=args.projector_learning_rate, momentum=0.9)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in decoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in decoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    t_total = args.steps
    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    scheduler_projector = get_linear_schedule_with_warmup(
        optimizer_projector, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    global_step = 0
    losses = []
    decoder.train()
    model.eval()
    projector.train()
    for i in range(args.steps):
        for batch_idx, (labels, _, toks, mask, type_id) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(data_loader)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            with torch.no_grad():
                out = model(toks, repr_layers=repr_layers) # ESM-2
            
            labels_mask_length = (labels.sum(0) > 0).sum()
            # print(labels_mask_length)
            labels = labels[:, torch.arange(labels_mask_length + 1)]
            encoder_hidden_states = projector(out["representations"][repr_layers[-1]])
            mask = mask[:, torch.arange(labels_mask_length + 1)]
            type_id = type_id[:, torch.arange(labels_mask_length + 1)]
            # encoder_hidden_states = None
            outputs = decoder(encoder_hidden_states=encoder_hidden_states, input_ids=labels, labels=labels, attention_mask=mask, token_type_ids=type_id)
            loss = outputs.loss
            if global_step % 4 == 0:
                logits = outputs.logits
                print(labels)
                print(torch.argmax(logits, 2))
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer_projector.step()
            scheduler.step()  # Update learning rate schedule
            scheduler_projector.step()
            decoder.zero_grad()
            projector.zero_grad()
            
            global_step += 1
            plt.plot(range(len(losses)), losses)
            plt.savefig(f"{args.output_name}.png")
            plt.close()
            if global_step > args.steps:
                break
        torch.save({'state_dict': decoder.state_dict(), 'steps': global_step, 'projector': projector.state_dict()}, args.output_name + ".pt") 
        if global_step > args.steps:
            break
    torch.save({'state_dict': decoder.state_dict(), 'steps': global_step, 'projector': projector.state_dict()}, args.output_name + ".pt") 
    
    

    
def main():
    parser = create_parser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output-name", type=str)
    parser.add_argument("--no-pretrained-encoder", action="store_true")
    args = parser.parse_args()
    run(args)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup_steps = args.warmup_steps if args.warmup_percent == 0 else int(args.warmup_percent*t_total)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(args.beta1,args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility

    best_auc = 0
    last_auc = 0
    stop_count = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in TOKEN_ID_GROUP else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)


                        if args.task_name == "dna690":
                            # record the best auc
                            if results["auc"] > best_auc:
                                best_auc = results["auc"]

                        if args.early_stop != 0:
                            # record current auc to perform early stop
                            if results["auc"] < last_auc:
                                stop_count += 1
                            else:
                                stop_count = 0

                            last_auc = results["auc"]
                            
                            if stop_count == args.early_stop:
                                logger.info("Early stop")
                                return global_step, tr_loss / global_step


                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.task_name == "dna690" and results["auc"] < best_auc:
                        continue
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    if args.task_name != "dna690":
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


if __name__ == "__main__":
    main()
