import sys
sys.path += ['../']
import torch
import os
from collections import defaultdict
import faiss
from utils.util import (
    barrier_array_merge,
    convert_to_string_id,
    is_first_worker,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data
)
import csv
import copy
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from data.msmarco_data import GetProcessingFn  
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import json
import logging
import random
import time
import pytrec_eval

torch.multiprocessing.set_sharing_strategy('file_system')


logger = logging.getLogger(__name__)


# ANN - active learning ------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    subdirectories = list(next(os.walk(args.training_dir))[1])

    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.training_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "scheduler.pt")
        return os.path.exists(scheduler_path)

    checkpoint_nums = [get_checkpoint_no(s) for s in subdirectories if valid_checkpoint(s)]
    if args.fix_refresh_rate > 0:
        checkpoint_nums = [x for x in checkpoint_nums if x % args.fix_refresh_rate == 0]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" +
                            str(max(checkpoint_nums))) + "/", max(checkpoint_nums)
    return args.init_model_dir, 0


def load_positive_ids(data_path, dev_set=False):

    logger.info(f"Loading query_2_pos_docid from {data_path}")
    query_positive_id = {}
    query_positive_id_path = os.path.join(
        data_path,
        "dev-qrel.tsv" if dev_set else "train-qrel.tsv"
    )
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)

            if not dev_set:
                assert rel == "1"
                query_positive_id[topicid] = docid
            else:        
                if topicid not in query_positive_id:
                    query_positive_id[topicid] = {}
                query_positive_id[topicid][docid] = max(0, int(rel))
    
    return query_positive_id


def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path
    config = configObj.config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="MSMarco",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        train_dataloader,
        is_query_inference=True,
        prefix=""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for idx, batch in enumerate(tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1,0],
                      position=0,
                      leave=True)):

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()}
            if is_query_inference:
                if args.world_size == 1:
                    embs = model.query_emb(**inputs)
                else:
                    embs = model.module.query_emb(**inputs)
            else:
                if args.world_size == 1:
                    embs = model.body_emb(**inputs)
                else:
                    embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, output_path, is_query_inference=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created

    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args, model, inference_dataloader, is_query_inference=is_query_inference, prefix=prefix)

    logger.info("merging embeddings")

    # preserve to memory
    full_embedding = barrier_array_merge(
        args,
        _embedding,
        prefix=prefix + "_emb_p_",
        output_path=output_path,
        load_cache=False,
        only_load_in_master=True)
    full_embedding2id = barrier_array_merge(
        args,
        _embedding2id,
        prefix=prefix + "_embid_p_",
        output_path=output_path,
        load_cache=False,
        only_load_in_master=True)

    return full_embedding, full_embedding2id


def generate_new_ann(
        args,
        output_num,
        checkpoint_path,
        srd_query_positive_id,
        srd_dev_query_positive_id,
        tgd_query_positive_id,
        latest_step_num):
    config, tokenizer, model = load_model(args, checkpoint_path)

    logger.info("***** inference of srd dev query *****")
    srd_dev_query_collection_path = os.path.join(args.srd_data_dir, "dev-query")
    srd_dev_query_cache = EmbeddingCache(srd_dev_query_collection_path)
    with srd_dev_query_cache as emb:
        srd_dev_query_embedding, srd_dev_query_embedding2id = StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=True),
            "dev_query_" + str(latest_step_num) + "_",
            emb,
            output_path=args.output_dir,
            is_query_inference=True
        )

    logger.info("***** inference of srd passages *****")
    srd_passage_collection_path = os.path.join(args.srd_data_dir, "passages")
    srd_passage_cache = EmbeddingCache(srd_passage_collection_path)
    with srd_passage_cache as emb:
        srd_passage_embedding, srd_passage_embedding2id = StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_" + str(latest_step_num) + "_",
            emb,
            output_path=args.output_dir,
            is_query_inference=False
        )

    if args.inference:
        return

    logger.info("***** inference of srd train query *****")
    srd_query_collection_path = os.path.join(args.srd_data_dir, "train-query")
    srd_query_cache = EmbeddingCache(srd_query_collection_path)
    with srd_query_cache as emb:
        srd_query_embedding, srd_query_embedding2id = StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=True),
            "query_" + str(latest_step_num) + "_",
            emb,
            output_path=args.output_dir,
            is_query_inference=True
        )
    
    if is_first_worker():
        # ANN search for dev passages and dev queries
        srd_dim = srd_passage_embedding.shape[1]
        print('srd passage embedding shape: ' + str(srd_passage_embedding.shape))
        faiss.omp_set_num_threads(16)
        srd_cpu_index = faiss.IndexFlatIP(srd_dim)
        srd_cpu_index.add(srd_passage_embedding)
        logger.info("***** Done Dev ANN Index *****")

        _, srd_I = srd_cpu_index.search(srd_dev_query_embedding, 100)  # I: [number of queries, topk]
        result_dict, num_queries_srd = EvalDevQuery(
            args, srd_dev_query_embedding2id, srd_passage_embedding2id,
            srd_dev_query_positive_id, srd_I)
        result_dict_with_srd_name = {}
        for k, v in result_dict.items():
            result_dict_with_srd_name['msmarco-'+k] = v
        dump_eval_result(result_dict_with_srd_name, args.output_dir, output_num, checkpoint_path)
    
    if args.tgd_data_dir is not None:
        logger.info("***** inference of tgd passages *****")
        tgd_passage_collection_path = os.path.join(args.tgd_data_dir, "passages")
        tgd_passage_cache = EmbeddingCache(tgd_passage_collection_path)
        with tgd_passage_cache as emb:
            tgd_passage_embedding, tgd_passage_embedding2id = StreamInferenceDoc(
                args,
                model,
                GetProcessingFn(args, query=False),
                "passage_" + str(latest_step_num) + "_",
                emb,
                output_path=args.tgd_output_dir,
                is_query_inference=False
            )
        
        logger.info("***** inference of tgd query *****")
        tgd_query_collection_path = os.path.join(args.tgd_data_dir, "train-query")
        tgd_query_cache = EmbeddingCache(tgd_query_collection_path)
        with tgd_query_cache as emb:
            tgd_query_embedding, tgd_query_embedding2id = StreamInferenceDoc(
                args,
                model,
                GetProcessingFn(args, query=True, tgd=True),
                "query_" + str(latest_step_num) + "_",
                emb,
                output_path=args.tgd_output_dir,
                is_query_inference=True
            )

    if is_first_worker():
        if args.tgd_data_dir is not None:
            construct_new_train_set(
                args,
                tgd_passage_embedding, tgd_passage_embedding2id,
                tgd_query_embedding, tgd_query_embedding2id,
                tgd_query_positive_id,
                output_num,
                checkpoint_path,
                output_path=args.tgd_output_dir
            )
        # the ranking training set: (query, pos_doc, [nearest_neg_doc]*n)
        construct_new_train_set(
            args,
            srd_passage_embedding, srd_passage_embedding2id,
            srd_query_embedding, srd_query_embedding2id,
            srd_query_positive_id,
            output_num,
            checkpoint_path,
            output_path=args.output_dir
        )

        # return result_dict['ndcg@20'], num_queries_dev


def dump_eval_result(result_dict, output_path, output_num, checkpoint_path):
    ndcg_output_path = os.path.join(
        output_path, f"ann_ndcg_" + str(output_num))
    if os.path.exists(ndcg_output_path):
        with open(ndcg_output_path) as fin:
            json_dict = json.load(fin)
    else:
        json_dict = {}
    json_dict.update(result_dict)
    with open(ndcg_output_path, 'w') as f:
        json_dict['checkpoint'] = checkpoint_path
        json.dump(json_dict, f)


def construct_new_nngan_train_set(
        args,
        srd_query_embedding, srd_query_embedding2id,
        srd_passage_embedding, srd_passage_embedding2id,
        tgd_passage_embedding, tgd_passage_embedding2id,
        output_num,
        checkpoint_path,
        output_path,
        max_size,
    ):

    # the domain adaptation training set:
    # THIS ONE NOT USED NOW: (srd_query, [nearest_srd_doc]*n, [nearest_tgd_doc]*n)
    # (srd_doc, [nearest_srd_doc]*n, [nearest_tgd_doc]*n)
    # (tgd_doc, [nearest_tgd_doc]*n, [nearest_srd_doc]*n)
    
    dim = srd_query_embedding.shape[1]
    faiss.omp_set_num_threads(16)

    srd_passage_index = faiss.IndexFlatIP(dim)
    srd_passage_index.add(srd_passage_embedding)
    tgd_passage_index = faiss.IndexFlatIP(dim)
    tgd_passage_index.add(tgd_passage_embedding)
    logger.info("***** Done srd & tgd passage index *****")
    
    chunk_factor = args.ann_chunk_factor
    effective_idx = output_num % chunk_factor
    if chunk_factor <= 0:
        chunk_factor = 1

    search_and_build_dataset(
        args,
        chunk_factor=chunk_factor, effective_idx=effective_idx,
        pos_index=srd_passage_index, pos_index2id=srd_passage_embedding2id,
        neg_index=tgd_passage_index, neg_index2id=tgd_passage_embedding2id,
        query_embedding=srd_passage_embedding, query_embedding2id=srd_passage_embedding2id,
        output_fname = os.path.join(output_path, f"sd_sd_td_{output_num}"),
        max_size=max_size
    )
    search_and_build_dataset(
        args,
        chunk_factor=1, effective_idx=0,
        pos_index=tgd_passage_index, pos_index2id=tgd_passage_embedding2id,
        neg_index=srd_passage_index, neg_index2id=srd_passage_embedding2id,
        query_embedding=tgd_passage_embedding, query_embedding2id=tgd_passage_embedding2id,
        output_fname = os.path.join(output_path, f"td_td_sd_{output_num}"),
        max_size=int(1e10)
    )


def search_and_build_dataset(
        args,
        chunk_factor, effective_idx,
        pos_index, pos_index2id,
        neg_index, neg_index2id,
        query_embedding, query_embedding2id,
        output_fname,
        max_size,
    ):
    
    num_queries = len(query_embedding)
    queries_per_chunk = num_queries // chunk_factor
    q_start_idx = queries_per_chunk * effective_idx
    if effective_idx == chunk_factor - 1:
        q_end_idx = num_queries
    else:
        q_end_idx = q_start_idx + queries_per_chunk
    q_end_idx = min(q_end_idx, q_start_idx+max_size)
    query_embedding = query_embedding[q_start_idx:q_end_idx]
    query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]
    effective_q_id = set(query_embedding2id.flatten())

    _, pos_I = pos_index.search(query_embedding, args.nn_topk_training)
    _, neg_I = neg_index.search(query_embedding, args.nn_posneg_sample)
    
    with open(output_fname, 'w') as fout:
        for query_idx in range(pos_I.shape[0]):
            if query_idx % 5000 == 0:
                logger.info(f"query_idx = {query_idx}")
            query_id = query_embedding2id[query_idx]
            if query_id not in effective_q_id:
                continue
            selected_pos_ann_idx = random.choices(
                pos_I[query_idx],  #[1:],  # excluding itself
                k=args.nn_posneg_sample
            )
            selected_neg_ann_idx = neg_I[query_idx]

            print("{}\t{}\t{}".format(
                query_id,
                ','.join([str(pos_index2id[pid]) for pid in selected_pos_ann_idx]),
                ','.join([str(neg_index2id[pid]) for pid in selected_neg_ann_idx]),
            ), file=fout)


def construct_new_train_set(
        args,
        passage_embedding, passage_embedding2id,
        query_embedding, query_embedding2id,
        training_query_positive_id,
        output_num,
        checkpoint_path,
        output_path,
    ):

    # ANN search for (train) passages and queries, output the new training set to files
    dim = passage_embedding.shape[1]
    print('passage embedding shape: ' + str(passage_embedding.shape))
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)
    logger.info("***** Done ANN Index *****")

    # Construct new training set ==================================
    chunk_factor = args.ann_chunk_factor
    effective_idx = output_num % chunk_factor

    if chunk_factor <= 0:
        chunk_factor = 1
    num_queries = len(query_embedding)
    queries_per_chunk = num_queries // chunk_factor
    q_start_idx = queries_per_chunk * effective_idx
    q_end_idx = num_queries if (
        effective_idx == (
            chunk_factor -
            1)) else (
        q_start_idx +
        queries_per_chunk)
    query_embedding = query_embedding[q_start_idx:q_end_idx]
    query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]

    logger.info(
        "Chunked {} query from {}".format(
            len(query_embedding),
            num_queries))
    # I: [number of queries, topk]
    _, I = cpu_index.search(query_embedding, args.topk_training)

    effective_q_id = set(query_embedding2id.flatten())
    query_negative_passage = GenerateNegativePassaageID(
        args,
        query_embedding2id,
        passage_embedding2id,
        training_query_positive_id,
        I,
        effective_q_id)

    logger.info("***** Construct ANN Triplet *****")
    train_data_output_path = os.path.join(
        output_path, f"ann_training_data_" + str(output_num))

    with open(train_data_output_path, 'w') as f:
        query_range = list(range(I.shape[0]))
        random.shuffle(query_range)
        for query_idx in query_range:
            query_id = query_embedding2id[query_idx]
            if query_id not in effective_q_id or query_id not in training_query_positive_id:
                continue
            pos_pid = training_query_positive_id[query_id]
            f.write(
                "{}\t{}\t{}\n".format(
                    query_id, pos_pid, ','.join(
                        str(neg_pid) for neg_pid in query_negative_passage[query_id])))


def GenerateNegativePassaageID(
        args,
        query_embedding2id,
        passage_embedding2id,
        training_query_positive_id,
        I_nearest_neighbor,
        effective_q_id):
    query_negative_passage = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]):

        query_id = query_embedding2id[query_idx]

        if query_id not in effective_q_id:
            continue

        num_queries += 1

        pos_pid = training_query_positive_id[query_id]
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        if SelectTopK:
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx)
            selected_ann_idx = top_ann_pid[negative_sample_I_idx]

        query_negative_passage[query_id] = []

        neg_cnt = 0
        rank = 0

        for idx in selected_ann_idx:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                if rank <= 10:
                    mrr += 1 / rank
                continue

            if neg_pid in query_negative_passage[query_id]:
                continue

            if neg_cnt >= args.negative_sample:
                break

            query_negative_passage[query_id].append(neg_pid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) +
              " --- ANN MRR:" + str(mrr / num_queries))

    return query_negative_passage


def EvalDevQuery(
        args,
        query_embedding2id,
        passage_embedding2id,
        dev_query_positive_id,
        I_nearest_neighbor):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {}

    for query_idx in range(I_nearest_neighbor.shape[0]):
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:50]
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id),
        {'map_cut', 'ndcg_cut', 'recip_rank','recall', 'P'}
    )

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))

    ndcg = defaultdict(int)
    precision = defaultdict(int)
    Map = 0
    mrr = 0
    recall_1k = 0
    recall_100 = 0
    cuts = [5, 10, 20]

    for k in result.keys():
        eval_query_cnt += 1
        for cut in cuts:
            ndcg[cut] += result[k][f"ndcg_cut_{cut}"]
            precision[cut] += result[k][f"P_{cut}"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall_1k += result[k]["recall_1000"]
        recall_100 += result[k]["recall_100"]

    result_dict = {}
    for cut in cuts:
        result_dict[f'ndcg@{cut}'] = ndcg[cut] / eval_query_cnt
        result_dict[f'p@{cut}'] = precision[cut] / eval_query_cnt
    result_dict['map'] = Map / eval_query_cnt
    result_dict['mrr'] = mrr / eval_query_cnt
    result_dict['recall_1k'] = recall_1k / eval_query_cnt
    result_dict['recall_100'] = recall_100 / eval_query_cnt


    print("Rank:" + str(args.rank) \
        + " --- ANN NDCG@20:" + str(result_dict['ndcg@20']) \
        + " --- ANN MRR:" + str(result_dict['mrr']) \
        + " --- ANN P@20:" + str(result_dict['p@20'])
    )

    return result_dict, eval_query_cnt


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--srd_data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir for source domain train set. "
             "Should contain the .tsv files (or other data files) for the task. "
             "For now it's msmarco.",
    )
    
    parser.add_argument(
        "--tgd_data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir for target domain train set. For now it's cqgtc.",
    )

    parser.add_argument(
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )

    parser.add_argument(
        "--init_model_dir",
        default=None,
        type=str,
        required=True,
        help="Initial model dir, will use this if no checkpoint is found in training_dir",
    )

    parser.add_argument(
        "--last_checkpoint_dir",
        default="",
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--tgd_output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the tgd data will be written",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )

    parser.add_argument(
        "--end_output_num",
        default=-
        1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
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
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default=5,  # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default=500,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default=5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--nn_topk_training",
        default=50,
        type=int,
        help="top k from which negative samples are collected (for nn discriminator)",
    )

    parser.add_argument(
        "--nn_posneg_sample",
        default=5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--fix_refresh_rate",
        type=int,
        default=0,
        help="Fix the ANN index refresh rate to X global steps. If X is 0 then we don't fix it.",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
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
    
    parser.add_argument(
        "--inference",
        default=False,
        action="store_true",
        help="only do inference if specify",
    )

    args = parser.parse_args()

    return args


def set_env(args):
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

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    else:
        args.world_size = 1
        args.rank = 0

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )


def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    ann_no, _, _ = get_latest_ann_data(args.output_dir)  # train only, since we only care about ann_no
    output_num = ann_no + 1

    logger.info("starting output number %d", output_num)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if args.tgd_output_dir is not None:
            if not os.path.exists(args.tgd_output_dir):
                os.makedirs(args.tgd_output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

    srd_positive_id = load_positive_ids(args.srd_data_dir)
    srd_dev_positive_id = load_positive_ids(args.srd_data_dir, dev_set=True)
    tgd_positive_id = None
    if args.tgd_data_dir is not None:
        tgd_positive_id = load_positive_ids(args.tgd_data_dir)

    while args.end_output_num == -1 or output_num <= args.end_output_num:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)

        if args.only_keep_latest_embedding_file:
            latest_step_num = 0

        if next_checkpoint == last_checkpoint:
            time.sleep(60)
        else:
            logger.info("start generate ann data number %d", output_num)
            logger.info("next checkpoint at " + next_checkpoint)
            generate_new_ann(  # for both train and tgd
                args,
                output_num,
                next_checkpoint,
                srd_positive_id,
                srd_dev_positive_id,
                tgd_positive_id,
                latest_step_num)
            if args.inference:
                break
            logger.info("finished generating ann data number %d", output_num)
            output_num += 1
            last_checkpoint = next_checkpoint
        if args.local_rank != -1:
            dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    main()
