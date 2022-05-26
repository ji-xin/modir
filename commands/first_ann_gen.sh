#!/bin/bash


pretrained_checkpoint_dir="../checkpoints/ANCE-passage"
srd_preprocessed_data_dir="../data/msmarco/preprocessed_data"
srd_first_ann_data_dir="../srd_first_ann_data"
checkpoint_dir="../saved_models"
mkdir -p $checkpoint_dir


N_FIRST_ANN_GPU=4
FIRST_ANN_GPU="0,1,2,3"

if [[ $N_FIRST_ANN_GPU = 0 ]]; then
  FIRST_ANN_LAUNCH=""
else
  FIRST_ANN_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_FIRST_ANN_GPU} --master_port 11211"
fi

first_ann_gen_cmd="\
CUDA_VISIBLE_DEVICES=${FIRST_ANN_GPU} python -u \
${FIRST_ANN_LAUNCH} ../drivers/run_ann_data_gen.py \
--training_dir ${checkpoint_dir} \
--init_model_dir ${pretrained_checkpoint_dir} \
--model_type rdot_nll \
--output_dir ${srd_first_ann_data_dir} \
--cache_dir "${srd_first_ann_data_dir}/cache/" \
--srd_data_dir ${srd_preprocessed_data_dir} \
--max_seq_length 512 \
--per_gpu_eval_batch_size 64 \
--topk_training 200 \
--negative_sample 20 \
--end_output_num 0"

echo $first_ann_gen_cmd
eval $first_ann_gen_cmd
