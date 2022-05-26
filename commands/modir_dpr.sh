#!/bin/bash

# hyperparameters
per_gpu_train_batch_size=8
per_gpu_eval_batch_size=64
gradient_accumulation_steps=1
WARMUP_STEPS=0
EVAL_STEPS=2500
LOG_STEPS=100
learning_rate=5e-6
dc_learning_rate=3e-5
lamb=5e-1


#############################################################
tgd_dataset="treccovid"

pretrained_checkpoint_dir="../checkpoints/ANCE-warmup"
srd_data_dir="../data/msmarco/raw_data"
tgd_data_dir="../data/${tgd_dataset}"
tinysrd_preprocessed_data_dir="../data/tinymsmarco/preprocessed_data"
tgd_preprocessed_data_dir="../data/${tgd_dataset}/preprocessed_data"

tsb_log_dir="../tsb_log/modir-dpr"
saved_models_dir="../saved_models"
saved_embedding_dir="../saved_embedding"
mkdir -p $saved_models_dir
mkdir -p $saved_embedding_dir




N_TRAIN_GPU=4
TRAIN_GPU="0,1,2,3"

if [[ $N_TRAIN_GPU = 0 ]]; then
  TRAIN_LAUNCH=""
else
  TRAIN_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_TRAIN_GPU} --master_port 11211"
fi


############################################# Training ########################################



cmd="CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python -u \
  ${TRAIN_LAUNCH} ../drivers/run_warmup.py \
  --train_model_type rdot_nll \
  --fp16 \
  --model_name_or_path ${pretrained_checkpoint_dir} \
  --task_name MSMarco \
  --do_train \
  --evaluate_during_training \
  --data_dir ${srd_data_dir} \
  --tgd_data_name ${tgd_dataset} \
  --tgd_data_dir ${tgd_data_dir} \
  --intraindev_data_name tinymsmarco,${tgd_dataset} \
  --intraindev_data_dir ${tinysrd_preprocessed_data_dir},${tgd_preprocessed_data_dir} \
  --max_seq_length 512 \
  --dropout_rate 0.0 \
  --per_gpu_eval_batch_size ${per_gpu_eval_batch_size} \
  --per_gpu_train_batch_size ${per_gpu_train_batch_size} \
  --learning_rate ${learning_rate} \
  --dc_learning_rate ${dc_learning_rate} \
  --num_train_epochs 2.0 \
  --output_dir ${saved_models_dir} \
  --saved_embedding_dir $saved_embedding_dir \
  --log_dir ${tsb_log_dir} \
  --warmup_steps ${WARMUP_STEPS} \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --expected_train_size 35000000 \
  --logging_steps ${LOG_STEPS} \
  --eval_steps ${EVAL_STEPS} \
  --dc_rep_steps 1000 \
  --dc_rep_method async \
  --dc_rep_step_per_batch 50 \
  --dc_loss_choice confusion \
  --lamb $lamb \
  --lamb_reduce_to_half_steps 10000 \
"


echo $cmd
eval $cmd
