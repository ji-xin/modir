import sys
sys.path += ['../']
import os
import time
import torch
from data.msmarco_data import GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn
from utils.util import (
    getattr_recursive,
    set_seed,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data,
    is_first_worker
)
import pandas as pd
from transformers import glue_processors as processors
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
import transformers
from utils.lamb import Lamb
from utils.modir_utils import (
    compute_total_grad_L2_norm, intrain_dev_eval, intrain_save_checkpoint,
    build_dl_iter_from_file, get_next,
    build_input_from_batch, get_module
)
from data.msmarco_data import GetProcessingFn
from model.models import MSMarcoConfigDict, ALL_MODELS
from model.domain_classifier import DomainClassifier, DummyModule, dry_dc_evaluation
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
import faiss
try:
    from apex import amp
except ImportError:
    print("apex not imported")
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)
# logging.disable(20)  # supressing logger.info
faiss.omp_set_num_threads(16)
faiss_dim = 768  # it's unlikely that this will need to be changed


def GetTripletTrainingDataProcessingFnWithSeparatePassageCache(
        args, query_cache, passage_cache, another_passage_cache
    ):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])

        for pos_pid, neg_pid in zip(
            [int(pos_pid) for pos_pid in line_arr[1].split(',')],
            [int(neg_pid) for neg_pid in line_arr[2].split(',')]
        ):
            query_data = GetProcessingFn(
                args, query=False)(
                query_cache[qid], qid)[0]
            pos_data = GetProcessingFn(
                args, query=False)(
                passage_cache[pos_pid], pos_pid)[0]
            neg_data = GetProcessingFn(
                args, query=False)(
                another_passage_cache[neg_pid], neg_pid)[0]

            yield (
                query_data[0], query_data[1], query_data[2],
                pos_data[0], pos_data[1], pos_data[2],
                neg_data[0], neg_data[1], neg_data[2]
            )

    return fn


def build_train_dataset_from_ann(
        args,
        query_cache, passage_cache,
        tb_writer,
        global_step,
        last_ann_no,
        ann_dir,
    ):
    # check if new ann training data is availabe
    ann_no, ann_path, ndcg_json = get_latest_ann_data(ann_dir)

    if ann_path is not None and ann_no != last_ann_no:
        try:
            logger.info("Training on new ANN data at %s", ann_path)
            with open(ann_path, 'r') as f:
                ann_training_data = f.readlines()

            aligned_size = (len(ann_training_data) //
                            args.world_size) * args.world_size
            ann_training_data = ann_training_data[:aligned_size]

            logger.info("Total ann queries: %d", len(ann_training_data))
            if args.triplet:
                train_dataset = StreamingDataset(
                    ann_training_data,
                    GetTripletTrainingDataProcessingFn(
                        args, query_cache, passage_cache, tgd=not(ann_dir==args.ann_dir))
                )
            else:
                train_dataset = StreamingDataset(
                    ann_training_data,
                    GetTrainingDataProcessingFn(
                        args, query_cache, passage_cache)
                )
            train_dataloader = DataLoader(
                train_dataset, batch_size=args.train_batch_size)

            if args.local_rank != -1:
                dist.barrier()
            update = True
        except FileNotFoundError:
            update = False
            train_dataloader = None
        
        if is_first_worker() and ann_dir==args.ann_dir:
            # add ndcg at checkpoint step used instead of current step
            # ndcg_json will not be None since this is args.ann_dir
            metric_step = ndcg_json['checkpoint'].strip('/').split('/')[-1].split('-')[-1]
            try:
                metric_step = int(metric_step)
            except ValueError:
                metric_step = 0
            for key in ndcg_json:
                if key != 'checkpoint':
                    tb_writer.add_scalar(
                        key, ndcg_json[key], metric_step
                    )

        last_ann_no = ann_no
        return update, (train_dataloader, last_ann_no)

    return False, (None, None)

    
def show(model):
    # for debugging: print the first parameter of the model
    for p in model.parameters():
        entry = p
        while True:
            try:
                entry = entry[0]
            except:
                return entry.item()


def train(args, model, dc_model, tokenizer,
          caches, tgd_file_name, file_process_fn,
):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    
    query_cache, passage_cache = caches
    tgd_file = open(tgd_file_name)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    # Create a static copy of dc_model
    static_dc_model = DomainClassifier(args)
    static_dc_model.to(args.device)

    # optimizer for ANCE
    optimizer_grouped_parameters = []
    layer_optim_params = set()
    for layer_name in [
        "roberta.embeddings",
        "score_out",
        "downsample1",
        "downsample2",
        "downsample3"]:
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

    if args.optimizer.lower() == "lamb":
        optimizer_constructor = lambda param, lr, decay: Lamb(
            param, lr=lr, eps=args.adam_epsilon, weight_decay=decay
        )
    elif args.optimizer.lower() == "adamw":
        optimizer_constructor = lambda param, lr, decay: AdamW(
            param, lr=lr, eps=args.adam_epsilon, weight_decay=decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer {args.optimizer} not recognized! Can only be lamb or adamW")
    
    optimizer = optimizer_constructor(optimizer_grouped_parameters, args.learning_rate, args.weight_decay)
    dc_optimizer = optimizer_constructor(dc_model.parameters(), args.dc_learning_rate, args.dc_weightDecay)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(
            args.model_name_or_path,
            "optimizer.pt")) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    args.model_name_or_path,
                    "optimizer.pt")))

    logger.info("Start fp16 and distributed model init")
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
        static_dc_model = torch.nn.DataParallel(static_dc_model)

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
            find_unused_parameters=True,
        )
        static_dc_model = torch.nn.parallel.DistributedDataParallel(
            static_dc_model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", args.max_steps)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(
        "  Gradient Accumulation steps = %d",
        args.gradient_accumulation_steps)

    global_step = 0
    dyn_lamb = args.lamb

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model
        # path
        if "-" in args.model_name_or_path:
            try:
                global_step = int(
                    args.model_name_or_path.split("-")[-1].split("/")[0])
            except:
                global_step=0
        else:
            global_step = 0
        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from global step %d", global_step)

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
    set_seed(args)  # Added here for reproductibility


    last_ann_no = -1
    train_dataloader = None
    train_dataloader_iter = None
    epoch_num = -1
    step = 0
    accumulated_srd_embs = []
    accumulated_tgd_embs = []
    prev_dry_dc_state_dict = None

    # actual_refresh_rate = None
    # prev_refresh_gstep = None
    # half_eval_done = False

    if args.single_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps)
        dc_scheduler = get_linear_schedule_with_warmup(
            dc_optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps)
    
    srd_update_ann, tgd_update_ann = False, False

    while global_step < args.max_steps:
        if step % args.gradient_accumulation_steps == 0:
            if global_step % args.logging_steps == 0:
                if not srd_update_ann:
                    srd_update_ann, (newdataloader, newlan) = build_train_dataset_from_ann(
                        args,
                        query_cache, passage_cache,
                        tb_writer,
                        global_step,
                        last_ann_no,
                        ann_dir=args.ann_dir
                    )
                
                # only update if both domains' new ann are ready
                if srd_update_ann:
                    train_dataloader, last_ann_no = newdataloader, newlan
                    train_dataloader_iter = iter(train_dataloader)
                    epoch_num += 1
                    if is_first_worker():
                        tb_writer.add_scalar(
                            'epoch', epoch_num, global_step
                        )

                    if global_step > 0:
                        prev_dry_dc_state_dict = intrain_dev_eval(
                            args, global_step, model, tb_writer, prev_dry_dc_state_dict,
                            all_datasets=True)
                        intrain_save_checkpoint(
                            args, global_step, model, tokenizer, optimizer, scheduler)
                    
                    srd_update_ann, tgd_update_ann = False, False
                
                _, tgd_epoch_iter = build_dl_iter_from_file(args, tgd_file, file_process_fn)

        step += 1

        # get srd batch and inputs
        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            logger.info("Finished iterating current dataset, begin reiterate")
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        batch = tuple(t.to(args.device) for t in batch)
        batch_size = batch[0].shape[0]
        inputs = build_input_from_batch(args, batch, mode='full', triplet=True)

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
            with model.no_sync(), dc_model.no_sync(), static_dc_model.no_sync():
                outputs = model(**inputs, output_dc_emb=True)
        
        ranking_loss = outputs[0]
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

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()
    
    tgd_file.close()

    return global_step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    
    parser.add_argument(
        "--tgd_raw_data_dir",
        default=None,
        type=str,
        help="The input raw data dir for target domain.",
    )
    
    parser.add_argument(
        "--tgd_data_name",
        default=None,
        type=str,
        required=False,
        help="The target domain dataset name.",
    )

    parser.add_argument(
        "--intraindev_data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir for in-train-dev set.",
    )
    parser.add_argument(
        "--intraindev_data_name",
        default=None,
        type=str,
        required=False,
        help="The in-train-dev dataset name.",
    )

    parser.add_argument(
        "--ann_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
    )

    parser.add_argument(
        "--tgd_ann_dir",
        default=None,
        type=str,
        required=False,
        help="The ann training data dir for tgd. Should contain the output of ann data generation job",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(
            MSMarcoConfigDict.keys()),
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(
            processors.keys()),
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
        help="The maximum total input sequence (document) length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input query length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length_tgd",
        default=None,
        type=int,
        help="The maximum total input query length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded. Default is args.max_query_length. "
        "This argument is used only when target domain is arguana, where max_query_len=512 is needed.",
    )

    parser.add_argument(
        "--triplet",
        default=False,
        action="store_true",
        help="Whether to run training.",
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
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--dc_method",
        default="classification",
        type=str,
        help="What to do for domain confusion. "
             "classification: classify the source of a vector representation. "
             "knn: a representation's k-nearest neighbors should have members from both domain "
             "(its implementation is removed; check c1fae2c).")

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
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--dc_learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for dc_model.",
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
        "--lamb",
        default=0.1,
        type=float,
        help="Coefficient for domain classification loss.",
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
        "--dc_weightDecay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some for domain classifier.",
    )

    parser.add_argument(
        "--no_gn_clip",
        action="store_true",
        help="Whether to disable grad norm clipping",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--max_steps",
        default=1000000,
        type=int,
        help="If > 0: set total number of training steps to perform",
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
        help="Evaluate the model every X updates steps.",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
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

    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )

    # ----------------- End of Doc Ranking HyperParam ------------------
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
    try:
        tgd_position = args.intraindev_data_name.index(args.tgd_data_name)
        args.intraindev_data_name[1], args.intraindev_data_name[tgd_position] = args.intraindev_data_name[tgd_position], args.intraindev_data_name[1]
        args.intraindev_data_dir[1], args.intraindev_data_dir[tgd_position] = args.intraindev_data_dir[tgd_position], args.intraindev_data_dir[1]
        args.mix_tgd = False
    except ValueError:
        args.mix_tgd = True
    
    return args


def set_env(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(
                args.server_ip,
                args.server_port),
            redirect_output=True)
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


def load_model(args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    else:
        args.world_size = 1

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    config = configObj.config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.output_hidden_states = True
    change_dropout_rate(config, args.dropout_rate)
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
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    def file_process_fn(line, i):
        return configObj.process_fn(line, i, tokenizer, args)

    return tokenizer, model, file_process_fn


def change_dropout_rate(config, val):
    config.attention_probs_dropout_prob = val
    config.hidden_dropout_prob = val


def save_checkpoint(args, model, tokenizer):
    # Saving best-practices: if you use defaults names for the model, you can
    # reload it using from_pretrained()
    if is_first_worker():
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

        # Good practice: save your training arguments together with the trained
        # model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.local_rank != -1:
        dist.barrier()


def main(profiling=False):
    args = get_arguments()
    if profiling:
        args.max_steps = 200
    set_env(args)
    tokenizer, model, file_process_fn = load_model(args)
    dc_model = DomainClassifier(args)
    dc_model.to(args.device)

    query_collection_path = os.path.join(args.data_dir, "train-query")
    query_cache = EmbeddingCache(query_collection_path)
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    
    tgd_file_name = os.path.join(args.tgd_raw_data_dir, "triples.simple.tsv")

    with query_cache, passage_cache:
        global_step = train(
            args, model, dc_model, tokenizer,
            (query_cache, passage_cache),
            tgd_file_name, file_process_fn
        )
        logger.info(" global_step = %s", global_step)

    save_checkpoint(args, model, tokenizer)


if __name__ == "__main__":
    profiling = False
    if profiling:
        import cProfile
        from pstats import SortKey
        cProfile.run("main(profiling=True)", sort=SortKey.CUMULATIVE)
    else:
        main()
