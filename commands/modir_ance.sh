#!/bin/bash

# hyperparameters
per_gpu_train_batch_size=4
per_gpu_eval_batch_size=64
gradient_accumulation_steps=4
WARMUP_STEPS=5000
EVAL_STEPS=2500
LOG_STEPS=100
learning_rate=1e-6
dc_learning_rate=5e-6
lamb=1.0


#############################################################
tgd_dataset="treccovid"

pretrained_checkpoint_dir="../checkpoints/ANCE-passage"
tgd_raw_data_dir="../data/${tgd_dataset}"
srd_preprocessed_data_dir="../data/msmarco/preprocessed_data"
tinysrd_preprocessed_data_dir="../data/tinymsmarco/preprocessed_data"
tgd_preprocessed_data_dir="../data/${tgd_dataset}/preprocessed_data"
srd_first_ann_data_dir="../srd_first_ann_data"
srd_ann_data_dir="../srd_ann_data"
mkdir -p ${srd_ann_data_dir}
rm -rf ${srd_ann_data_dir}/*
cp ${srd_first_ann_data_dir}/ann* ${srd_ann_data_dir}

tsb_log_dir="../tsb_log/modir-ance"
saved_models_dir="../saved_models"
saved_embedding_dir="../saved_embedding"
mkdir -p $saved_models_dir
mkdir -p $saved_embedding_dir
rm -f ${tsb_log_dir}/*



TRAIN_GPU="0,1"
INF_GPU="2,3"
N_TRAIN_GPU=2
N_INF_GPU=2

if [[ $N_TRAIN_GPU = 0 ]]; then
  TRAIN_LAUNCH=""
else
  TRAIN_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_TRAIN_GPU} --master_port 11211"
fi
if [[ $N_INF_GPU = 0 ]]; then
  INF_LAUNCH=""
else
  INF_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_INF_GPU} --master_port 21211"
fi


##################################### ANN Data generation ################################

ann_data_gen_cmd="\
CUDA_VISIBLE_DEVICES=${INF_GPU} python -u \
${INF_LAUNCH} ../drivers/run_ann_data_gen.py \
--training_dir ${saved_models_dir} \
--init_model_dir ${pretrained_checkpoint_dir} \
--model_type rdot_nll \
--output_dir ${srd_ann_data_dir} \
--cache_dir "${srd_ann_data_dir}/cache/" \
--srd_data_dir ${srd_preprocessed_data_dir} \
--max_seq_length 512 \
--per_gpu_eval_batch_size ${per_gpu_eval_batch_size} \
--topk_training 200 \
--negative_sample 20 \
&>../gen.log"

echo $ann_data_gen_cmd
eval $ann_data_gen_cmd &

############################################# Training ########################################


train_cmd="\
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python -u \
${TRAIN_LAUNCH} ../drivers/run_ann.py \
--model_type rdot_nll \
--fp16 \
--model_name_or_path ${pretrained_checkpoint_dir} \
--task_name MSMarco \
--triplet \
--data_dir ${srd_preprocessed_data_dir} \
--tgd_raw_data_dir ${tgd_raw_data_dir} \
--intraindev_data_name tinymsmarco,${tgd_dataset} \
--intraindev_data_dir ${tinysrd_preprocessed_data_dir},${tgd_preprocessed_data_dir} \
--ann_dir ${srd_ann_data_dir} \
--tgd_data_name ${tgd_dataset} \
--saved_embedding_dir ${saved_embedding_dir} \
--max_seq_length 512 \
--dropout_rate 0.0 \
--per_gpu_train_batch_size ${per_gpu_train_batch_size} \
--per_gpu_eval_batch_size ${per_gpu_eval_batch_size} \
--gradient_accumulation_steps ${gradient_accumulation_steps} \
--learning_rate ${learning_rate} \
--dc_learning_rate ${dc_learning_rate} \
--output_dir ${saved_models_dir} \
--log_dir ${tsb_log_dir} \
--warmup_steps ${WARMUP_STEPS} \
--logging_steps ${LOG_STEPS} \
--eval_steps ${EVAL_STEPS} \
--optimizer lamb \
--single_warmup \
--dc_rep_steps 1000 \
--dc_rep_method async \
--dc_rep_step_per_batch 50 \
--dc_loss_choice confusion \
--lamb ${lamb} \
--lamb_reduce_to_half_steps 10000 \
"

echo $train_cmd 
eval $train_cmd

wait
