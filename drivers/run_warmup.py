import sys
sys.path += ["../"]
import pandas as pd
from transformers import glue_compute_metrics as compute_metrics, glue_output_modes as output_modes, glue_processors as processors
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
import transformers
from utils.eval_mrr import passage_dist_eval
from model.models import MSMarcoConfigDict
from model.domain_classifier import DomainClassifier, DummyModule
from utils.lamb import Lamb
from utils.modir_utils import compute_total_grad_L2_norm, intrain_dev_eval, intrain_save_checkpoint
import os
from os import listdir
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
import numpy as np
import torch
from tqdm import tqdm, trange
import torch.distributed as dist
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch import nn
from utils.util import getattr_recursive, set_seed, is_first_worker
from utils.modir_utils import (
    compute_total_grad_L2_norm, intrain_dev_eval, intrain_save_checkpoint,
    build_dl_iter_from_file, get_next,
    build_input_from_batch, get_module
)

try:
    from apex import amp
except ImportError:
    print("apex not imported")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def train(args, model, dc_model, tokenizer, train_file, tgd_file, file_process_fn):
    """ Train the model """
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Create a static copy of dc_model
    static_dc_model = DomainClassifier(args)
    static_dc_model.to(args.device)

    if args.max_steps > 0:
        t_total = args.max_steps
    else:
        t_total = args.expected_train_size // real_batch_size * args.num_train_epochs

    # layerwise optimization for lamb
    optimizer_grouped_parameters = []
    layer_optim_params = set()
    for layer_name in ["roberta.embeddings", "score_out", "downsample1", "downsample2", "downsample3", "embeddingHead"]:
        layer = getattr_recursive(model, layer_name)
        if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    if getattr_recursive(model, "roberta.encoder.layer") is not None:
        for layer in model.roberta.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    optimizer_grouped_parameters.append(
        {"params": [p for p in model.parameters() if p not in layer_optim_params]})

    optimizer_constructors = {
        "lamb": lambda param, lr: Lamb(
            param, lr=lr, eps=args.adam_epsilon
        ),
        "adamw": lambda param, lr: AdamW(
            param, lr=lr, eps=args.adam_epsilon
        ),
        "sgd": lambda param, lr: SGD(
            param, lr=lr,
        )
    }
    
    optimizer = optimizer_constructors[args.optimizer.lower()](
        optimizer_grouped_parameters, args.learning_rate)
    dc_optimizer = optimizer_constructors[args.dc_optimizer.lower()](
        dc_model.parameters(), args.dc_learning_rate)

    if args.scheduler.lower() == "linear":
        print('Total steps', t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        dc_scheduler = get_linear_schedule_with_warmup(
            dc_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    elif args.scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, t_total, 1e-8)
        dc_scheduler = CosineAnnealingLR(dc_optimizer, t_total, 1e-8)
    elif args.scheduler.lower() == "step":
        # reduce learning rate by a half every 50k steps
        scheduler = StepLR(optimizer, step_size=50000, gamma=0.5)
        dc_scheduler = StepLR(dc_optimizer, step_size=50000, gamma=0.5)
    else:
        raise Exception(
            "Scheduler {0} not recognized!".format(args.scheduler))

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        if 'apex' not in sys.modules:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        [model, dc_model, static_dc_model], [optimizer, dc_optimizer] = amp.initialize(
            [model, dc_model, static_dc_model],
            [optimizer, dc_optimizer],
            opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        dc_model = torch.nn.DataParallel(dc_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        dc_model = torch.nn.parallel.DistributedDataParallel(
            dc_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )
        static_dc_model = torch.nn.parallel.DistributedDataParallel(
            static_dc_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    dyn_lamb = args.lamb  # dynamic lamb, the lamb that's actually used

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (args.expected_train_size //
                                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                args.expected_train_size // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info(
                "  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)
        except:
            logger.info("  Start training from a pretrained model")

    tr_loss = 0  # useless but just keep it
    optim_monitors = [
        'loss_adv_D', 'loss_adv_M', 'loss_ranking',
        'dc_total_Q', 'dc_correct_Q', 'dc_total_P', 'dc_correct_P',
        'dc_pre_softmax_logits_0', 'dc_pre_softmax_logits_1',
        'dc_post_softmax_prob_0', 'dc_post_softmax_prob_1',
        'embedding_norm',
    ]
    optim_cumulator = {k: 0.0 for k in optim_monitors}

    model_parts = ['roberta', 'projection']
    model_parts_params = {
        'roberta': [p for n, p in model.named_parameters() if 'embeddingHead' not in n],
        'projection': [p for n, p in model.named_parameters() if 'embeddingHead' in n],
        # 'domain_classifier': dc_model.parameters(),
    }
    grad_norm_cumulator = {k: 0.0 for k in model_parts}
    grad_norm_cumulator.update({k+'-clipped': 0.0 for k in model_parts})
    grad_norm_cumulator.update({
        'domain_classifier': 0.0, 'domain_classifier-clipped': 0.0
    })

    model.zero_grad()
    model.train()
    dc_model.zero_grad()
    dc_model.train()
    tqdm_disable = True
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
        disable=tqdm_disable or args.local_rank not in [-1, 0],
    )  # each iter is 1 epoch
    set_seed(args)  # Added here for reproductibility
    accumulated_srd_embs = []
    accumulated_tgd_embs = []
    prev_dry_dc_state_dict = None


    for m_epoch in train_iterator:

        if is_first_worker():
            tb_writer.add_scalar(
                'epoch', m_epoch, global_step
            )

        # get srd and tgd batches
        epoch_dataloader, _ = build_dl_iter_from_file(args, train_file, file_process_fn)
        _, tgd_epoch_iter = build_dl_iter_from_file(args, tgd_file, file_process_fn)
        
        for step, batch in tqdm(
            enumerate(epoch_dataloader), desc="Iteration",
            disable=tqdm_disable or args.local_rank not in [-1,0]
        ):
            model.train()  # ?

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # get srd batch and inputs
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            batch = tuple(t.to(args.device).long() for t in batch)
            batch_size = batch[0].shape[0]
            inputs = build_input_from_batch(args, batch, mode='full')

            # get tgd batch and inputs
            tgd_batch, tgd_epoch_iter = get_next(
                tgd_epoch_iter, args, tgd_file, file_process_fn, batch_size)
            tgd_batch = tuple(t.to(args.device).long() for t in tgd_batch)
            tgd_query_inputs = build_input_from_batch(args, tgd_batch, mode='query')
            if step % 2 == 0:
                tgd_doc_inputs = build_input_from_batch(args, tgd_batch, mode='pos_doc')
            else:
                tgd_doc_inputs = build_input_from_batch(args, tgd_batch, mode='neg_doc')


            ##### 1. forward of the encoder model #####

            if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                outputs = model(**inputs, output_dc_emb=True)
            else:
                with model.no_sync():
                    outputs = model(**inputs, output_dc_emb=True)
            
            ranking_loss = outputs[0]  # ranking loss
            if step % 2 == 0:
                srd_embs = [outputs[1][0], outputs[1][1]]
            else:
                srd_embs = [outputs[1][0], outputs[1][2]]
            
            if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                tgd_query_emb = get_module(model).query_emb(**tgd_query_inputs)
                tgd_doc_emb = get_module(model).body_emb(**tgd_doc_inputs)
            else:
                with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                    tgd_query_emb = get_module(model).query_emb(**tgd_query_inputs)
                    tgd_doc_emb = get_module(model).body_emb(**tgd_doc_inputs)
            tgd_embs = [tgd_query_emb, tgd_doc_emb]

            detached_srd_embs = [torch.tensor(x) for x in srd_embs]
            detached_tgd_embs = [torch.tensor(x) for x in tgd_embs]
            if args.dc_rep_method == 'async':
                if len(accumulated_srd_embs) == args.dc_rep_steps:
                    accumulated_srd_embs.pop(0)
                    accumulated_tgd_embs.pop(0)
                accumulated_srd_embs.append(detached_srd_embs)
                accumulated_tgd_embs.append(detached_tgd_embs)
            for emb in srd_embs+tgd_embs:
                optim_cumulator['embedding_norm'] += emb.norm(dim=1).mean() / 4
            
            if args.n_gpu > 1:
                ranking_loss = ranking_loss.mean()
            if args.gradient_accumulation_steps > 1:
                ranking_loss = ranking_loss / args.gradient_accumulation_steps
            optim_cumulator['loss_ranking'] += ranking_loss.item()

            # 2. feed detached embeddings to the dc_model and BP L_adv_D


            for dc_rep_step in range(1+args.dc_rep_steps):

                if args.dc_rep_method == 'repeat':
                    srd_dc_input_embs = detached_srd_embs
                    tgd_dc_input_embs = detached_tgd_embs
                elif args.dc_rep_method == 'async':
                    which_step = min(dc_rep_step, len(accumulated_srd_embs)-1)
                    srd_dc_input_embs = accumulated_srd_embs[which_step]
                    tgd_dc_input_embs = accumulated_tgd_embs[which_step]

                if dc_rep_step == 0:
                    batched_srd_dc_input_embs = srd_dc_input_embs
                    batched_tgd_dc_input_embs = tgd_dc_input_embs
                elif dc_rep_step % args.dc_rep_step_per_batch != 0:
                    batched_srd_dc_input_embs[0].append(srd_dc_input_embs[0])
                    batched_srd_dc_input_embs[1].append(srd_dc_input_embs[1])
                    batched_tgd_dc_input_embs[0].append(tgd_dc_input_embs[0])
                    batched_tgd_dc_input_embs[1].append(tgd_dc_input_embs[1])
                    continue
                else:
                    batched_srd_dc_input_embs[0].append(srd_dc_input_embs[0])
                    batched_srd_dc_input_embs[1].append(srd_dc_input_embs[1])
                    batched_tgd_dc_input_embs[0].append(tgd_dc_input_embs[0])
                    batched_tgd_dc_input_embs[1].append(tgd_dc_input_embs[1])
                    batched_srd_dc_input_embs[0] = torch.cat(batched_srd_dc_input_embs[0])
                    batched_srd_dc_input_embs[1] = torch.cat(batched_srd_dc_input_embs[1])
                    batched_tgd_dc_input_embs[0] = torch.cat(batched_tgd_dc_input_embs[0])
                    batched_tgd_dc_input_embs[1] = torch.cat(batched_tgd_dc_input_embs[1])


                # 2.1 feed detached embeddings to the dc_model
                L_adv_D = 0.0
                label_size = batch_size * (1 if dc_rep_step==0 else args.dc_rep_step_per_batch)
                srd_labels = torch.tensor([0] * label_size, device=args.device)
                tgd_labels = torch.tensor([1] * label_size, device=args.device)

                for i_emb, emb in enumerate(batched_srd_dc_input_embs):
                    labels = srd_labels

                    if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                        dc_srd_outputs = dc_model(emb, labels=labels)
                    else:
                        with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                            dc_srd_outputs = dc_model(emb, labels=labels)
                    L_adv_D += dc_srd_outputs[1] * args.dc_rep_step_per_batch  # scale up because of the average in cross_entropy

                    if dc_rep_step == 0:
                        suffix = 'Q' if i_emb==0 else 'P'
                        optim_cumulator[f'dc_total_{suffix}'] += dc_srd_outputs[2][0]
                        optim_cumulator[f'dc_correct_{suffix}'] += dc_srd_outputs[2][1]
                        optim_cumulator['dc_pre_softmax_logits_0'] += dc_srd_outputs[0][:, 0].mean() / 4
                        optim_cumulator['dc_pre_softmax_logits_1'] += dc_srd_outputs[0][:, 1].mean() / 4
                        probs = torch.softmax(dc_srd_outputs[0], dim=1)
                        optim_cumulator['dc_post_softmax_prob_0'] += probs[:, 0].mean() / 4
                        optim_cumulator['dc_post_softmax_prob_1'] += probs[:, 1].mean() / 4
                
                for i_emb, emb in enumerate(batched_tgd_dc_input_embs):
                    labels = tgd_labels

                    if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                        dc_tgd_outputs = dc_model(emb, labels=labels)
                    else:
                        with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                            dc_tgd_outputs = dc_model(emb, labels=labels)
                    L_adv_D += dc_tgd_outputs[1] * args.dc_rep_step_per_batch  # scale up because of the average in cross_entropy

                    if dc_rep_step == 0:
                        suffix = 'Q' if i_emb==0 else 'P'
                        optim_cumulator[f'dc_total_{suffix}'] += dc_tgd_outputs[2][0]
                        optim_cumulator[f'dc_correct_{suffix}'] += dc_tgd_outputs[2][1]
                        optim_cumulator['dc_pre_softmax_logits_0'] += dc_tgd_outputs[0][:, 0].mean() / 4
                        optim_cumulator['dc_pre_softmax_logits_1'] += dc_tgd_outputs[0][:, 1].mean() / 4
                        probs = torch.softmax(dc_tgd_outputs[0], dim=1)
                        optim_cumulator['dc_post_softmax_prob_0'] += probs[:, 0].mean() / 4
                        optim_cumulator['dc_post_softmax_prob_1'] += probs[:, 1].mean() / 4
                
                if dc_rep_step % args.dc_rep_step_per_batch == 0:
                    batched_srd_dc_input_embs = [[], []]
                    batched_tgd_dc_input_embs = [[], []]
                if dc_rep_step == 0:
                    continue  # this dc_rep_step is only for logging things for optim_cumulator
                    
                if args.n_gpu > 1:
                    L_adv_D = L_adv_D.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    L_adv_D = L_adv_D / args.gradient_accumulation_steps
                optim_cumulator['loss_adv_D'] += L_adv_D.item() / args.dc_rep_steps

                # 2.2 BP of L_adv_D; dc_optimizer update
                if args.fp16:
                    with amp.scale_loss(L_adv_D, dc_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                        L_adv_D.backward()
                    else:
                        with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                            L_adv_D.backward()

                if step % args.gradient_accumulation_steps == 0:
                    grad_norm_cumulator['domain_classifier'] += compute_total_grad_L2_norm(
                        dc_model.parameters()
                    ) / args.dc_rep_steps
                    if not args.no_gn_clip:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(dc_optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                dc_model.parameters(), args.max_grad_norm)
                    grad_norm_cumulator['domain_classifier-clipped'] += compute_total_grad_L2_norm(
                        dc_model.parameters()
                    ) / args.dc_rep_steps
                    
                    dc_optimizer.step()
                    dc_model.zero_grad()

            if step % args.gradient_accumulation_steps == 0:
                dc_scheduler.step()  # this is outside of the dc_rep_step loop

            # 3.1 copy the dc_model, feed (undetached) embeddings to it
            get_module(static_dc_model).load_state_dict(get_module(dc_model).state_dict())

            L_adv_M = 0.0
            if args.dc_loss_choice == 'minimax':
                srd_labels = torch.tensor([0] * batch_size, device=args.device)
                tgd_labels = torch.tensor([1] * batch_size, device=args.device)
            elif args.dc_loss_choice == 'gan':
                tgd_labels = torch.tensor([0] * batch_size, device=args.device)
            elif args.dc_loss_choice == 'confusion':
                srd_labels = 'uniform'
                tgd_labels = 'uniform'
            else:
                raise NotImplementedError()
                    
            if args.dc_loss_choice != 'gan':
                for emb in srd_embs:
                    if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                        dc_srd_outputs = static_dc_model(emb, labels=srd_labels)
                    else:
                        with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                            dc_srd_outputs = static_dc_model(emb, labels=srd_labels)
                    L_adv_M += dc_srd_outputs[1]
            
            for emb in tgd_embs:
                if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                    dc_tgd_outputs = static_dc_model(emb, labels=tgd_labels)
                else:
                    with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                        dc_tgd_outputs = static_dc_model(emb, labels=tgd_labels)
                L_adv_M += dc_tgd_outputs[1]
            
            if args.dc_loss_choice == 'minimax':
                L_adv_M = -L_adv_M
            L_adv_M *= dyn_lamb
                
            if args.n_gpu > 1:
                L_adv_M = L_adv_M.mean()
            if args.gradient_accumulation_steps > 1:
                L_adv_M = L_adv_M / args.gradient_accumulation_steps
            optim_cumulator['loss_adv_M'] += L_adv_M.item()

            # 3.2 BP of ranking loss and L_adv_M; optimizer update
            loss = ranking_loss + L_adv_M
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                if step % args.gradient_accumulation_steps == 0 or args.world_size == 1:
                    loss.backward()
                else:
                    with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                        loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                for model_part, params in model_parts_params.items():
                    grad_norm_cumulator[model_part] += compute_total_grad_L2_norm(params)
                if not args.no_gn_clip:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                for model_part, params in model_parts_params.items():
                    grad_norm_cumulator[model_part+'-clipped'] += compute_total_grad_L2_norm(params)
                
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            # end of the main part of training


            if step % args.gradient_accumulation_steps == 0:

                if args.lamb_reduce_to_half_steps > 0:
                    if is_first_worker():
                        tb_writer.add_scalar("lambda", dyn_lamb, global_step)
                    dyn_lamb = args.lamb * 2**(-global_step / args.lamb_reduce_to_half_steps)


                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    logs["linear_layer_L2norm"] = get_module(dc_model).layers[0].weight.norm().item()
                    logs["linear_layer_mean"] = get_module(dc_model).layers[0].weight.mean().item()
                    logs["learning_rate"] = scheduler.get_last_lr()[0]
                    logs["learning_rate_dc"] = dc_optimizer.param_groups[0]['lr']
                    logs["dc_acc_Q"] = optim_cumulator['dc_correct_Q'] / (1e-10 + optim_cumulator['dc_total_Q'])
                    logs["dc_acc_P"] = optim_cumulator['dc_correct_P'] / (1e-10 + optim_cumulator['dc_total_P'])
                    for k in optim_monitors:
                        if k not in ['dc_total_Q', 'dc_correct_Q', 'dc_total_P', 'dc_correct_P']:
                            logs[k] = float(optim_cumulator[k] / args.logging_steps / args.gradient_accumulation_steps)
                    optim_cumulator = {k: 0.0 for k in optim_monitors}  # reset

                    if is_first_worker():
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logs.update({k: v/args.logging_steps for k, v in grad_norm_cumulator.items()})
                        logger.info(json.dumps({**logs, **{"step": global_step}}))
                        for key, value in grad_norm_cumulator.items():
                            tb_writer.add_scalar(
                                'grad_norm-'+key,
                                value / args.logging_steps,
                                global_step)
                            grad_norm_cumulator[key] = 0.0  # reset

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    prev_dry_dc_state_dict = intrain_dev_eval(
                        args, global_step, model, tb_writer, prev_dry_dc_state_dict)
                    intrain_save_checkpoint(
                        args, global_step, model, tokenizer, optimizer, scheduler)


        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step, tr_loss / global_step


def load_stuff(model_type, args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.num_labels = num_labels

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        print('world_size', args.world_size)
    else:
        args.world_size = 1

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    configObj = MSMarcoConfigDict[model_type]
    model_args = type('', (), {})()
    model_args.use_mean = configObj.use_mean
    config = configObj.config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.output_hidden_states = True
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        model_argobj=model_args,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return config, tokenizer, model, configObj


def get_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    
    parser.add_argument(
        "--tgd_data_name",
        default=None,
        type=str,
        required=False,
        help="The target domain dataset name; if there are multiple, separate with commas.",
    )

    parser.add_argument(
        "--tgd_data_dir",
        default=None,
        type=str,
        required=False,
        help="The target domain input data dir; if there are multiple, separate with commas.",
    )
    
    parser.add_argument(
        "--intraindev_data_dir",
        default=None,
        type=str,
        required=False,
        help="The dev set data dir; if there are multiple, separate with commas.",
    )
    
    parser.add_argument(
        "--intraindev_data_name",
        default=None,
        type=str,
        required=False,
        help="The dev set dataset name; if there are multiple, separate with commas.",
    )

    parser.add_argument(
        "--train_model_type",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", 
        default="", 
        type=str, 
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--saved_embedding_dir",
        default="",
        type=str,
        help="The directory where intraindev embeddings are dumped",
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", 
        action="store_true",
        help="Whether to run training.",
    )

    parser.add_argument(
        "--do_eval", 
        action="store_true",
        help="Whether to run eval on the dev set.",
    )

    parser.add_argument(
        "--evaluate_during_training", 
        action="store_true", 
        help="Rul evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--do_lower_case", 
        action="store_true", 
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--eval_type",
        default="full",
        type=str,
        help="MSMarco eval type - dev full or small",
    )

    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW or SGD",
    )

    parser.add_argument(
        "--dc_optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW or SGD",
    )

    parser.add_argument(
        "--scheduler",
        default="linear",
        type=str,
        help="Scheduler - linear, cosine, or step",
    )

    parser.add_argument(
        "--dc_loss_choice",
        default="minimax",
        type=str,
        help="Adversarial loss choice (ADDA paper, Table 1, 4th column).")

    parser.add_argument(
        "--dc_layers",
        default=1,
        type=int,
        help="How many layers to use for the domain classifier",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", 
        default=8, 
        type=int, 
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size", 
        default=8, 
        type=int, 
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate", 
        default=5e-5,
        type=float, 
        help="The initial learning rate for the ranker model.",
    )

    parser.add_argument(
        "--dc_learning_rate", 
        default=5e-5,
        type=float, 
        help="The initial learning rate for the domain classifier.",
    )

    parser.add_argument(
        "--lamb", 
        default=0.0,
        type=float, 
        help="HP for GAN loss.",
    )

    parser.add_argument(
        "--lamb_reduce_to_half_steps",
        default=0,
        type=int,
        help="Reduce dyn_lamb exponentially, and it will be reduced to a half after X steps.",
    )

    parser.add_argument(
        "--dc_rep_steps",
        default=1,
        type=int,
        help="Update dc_model over a single batch for X steps.",
    )

    parser.add_argument(
        "--dc_rep_method",
        default="repeat",
        type=str,
        help="Use what data for dc repetitive training. "
             "repeat: use the same batch repetitively; "
             "async: use embeddings recorded from previous batches."
    )

    parser.add_argument(
        "--dc_rep_step_per_batch",
        default=1,
        type=int,
        help="For dc_rep, how many steps of embeddings to put in one batch",
    )

    parser.add_argument(
        "--no_gn_clip",
        action="store_true",
        help="Whether to disable grad norm clipping",
    )

    parser.add_argument(
        "--weight_decay", 
        default=0.0,
        type=float, 
        help="Weight decay if we apply some.",
    )

    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout probability",
    )

    parser.add_argument(
        "--adam_epsilon", 
        default=1e-8,
        type=float, 
        help="Epsilon for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm", 
        default=1.0,
        type=float, 
        help="Max gradient norm.",
    )

    parser.add_argument(
        "--num_train_epochs", 
        default=3.0, 
        type=float, 
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--warmup_steps",
         default=0, 
         type=int,
         help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--logging_steps", 
        type=int,
        default=500, 
        help="Log every X updates steps.",
    )

    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=500,
        help="Evaluate and save checkpoint every X global steps.",
    )

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    parser.add_argument(
        "--no_cuda", 
        action="store_true",
        help="Avoid using CUDA when available",
    )

    parser.add_argument(
        "--overwrite_output_dir", 
        action="store_true", 
        help="Overwrite the content of the output directory",
    )

    parser.add_argument(
        "--overwrite_cache", 
        action="store_true", 
        help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="random seed for initialization",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--expected_train_size",
        default=100000,
        type=int,
        help="Expected train dataset size",
    )

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--server_ip", 
        type=str, 
        default="",
        help="For distant debugging.",
    )

    parser.add_argument(
        "--server_port", 
        type=str,
        default="", 
        help="For distant debugging.",
    )

    args = parser.parse_args()

    # sort intraindev datasets, so that tinymsmarco is the first and the target domain dataset is the second
    args.intraindev_data_name = args.intraindev_data_name.split(',')
    args.intraindev_data_dir = args.intraindev_data_dir.split(',')
    assert args.intraindev_data_name[0] == 'tinymsmarco'
    assert len(args.intraindev_data_name) >= 2
    tgd_position = args.intraindev_data_name.index(args.tgd_data_name)
    args.intraindev_data_name[1], args.intraindev_data_name[tgd_position] = args.intraindev_data_name[tgd_position], args.intraindev_data_name[1]
    args.intraindev_data_dir[1], args.intraindev_data_dir[tgd_position] = args.intraindev_data_dir[tgd_position], args.intraindev_data_dir[1]

    return args


def set_env(args):
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def save_checkpoint(args, model, tokenizer):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.local_rank != -1:
        dist.barrier()


def evaluation(args, model, tokenizer):
    # Evaluation
    results = {}
    if args.do_eval:
        model_dir = args.model_name_or_path if args.model_name_or_path else args.output_dir

        checkpoints = [model_dir]

        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split(
                "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.eval()
            reranking_mrr, full_ranking_mrr = passage_dist_eval(
                args, model, tokenizer)
            if is_first_worker():
                print(
                    "Reranking/Full ranking mrr: {0}/{1}".format(str(reranking_mrr), str(full_ranking_mrr)))
            if args.local_rank != -1:
                dist.barrier()
    return results


def main():
    args = get_arguments()
    set_env(args)

    config, tokenizer, model, configObj = load_stuff(
        args.train_model_type, args)
        
    dc_model = DomainClassifier(
        args,
        input_size=config.hidden_size,
        n_class=2
    )
    dc_model.to(args.device)

    # Training
    if args.do_train:
        logger.info("Training/evaluation parameters %s", args)

        def file_process_fn(line, i):
            return configObj.process_fn(line, i, tokenizer, args)

        train_fname = args.data_dir+"/triples.train.small.tsv"
        train_file = open(train_fname, encoding="utf-8-sig")
        tgd_file = open(os.path.join(args.tgd_data_dir, "triples.simple.tsv"))
        
        global_step, tr_loss = train(
            args, model, dc_model, tokenizer, train_file, tgd_file, file_process_fn)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        train_file.close()
        tgd_file.close()

    save_checkpoint(args, model, tokenizer)

    results = evaluation(args, model, tokenizer)
    return results


if __name__ == "__main__":
    main()
