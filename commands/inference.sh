#!/bin/bash

TASK=treccovid

query_length=64
if [[ $TASK = "arguana" ]]; then
  query_length=512
fi

preprocessed_data_dir="../data/${TASK}/preprocessed_data"
checkpoint_dir="../checkpoints/modir-treccovid-10k"

N_GPU=4  # 0 for using single gpu
INF_GPU="0,1,2,3"

if [[ $N_GPU = 0 ]]; then
  INF_LAUNCH=""
else
  INF_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port 11211"
fi


##################################### Inference ################################
output_dir="../inference_output"
mkdir -p ${output_dir}
rm -f ${output_dir}/*.pb

inf_cmd="\
CUDA_VISIBLE_DEVICES=${INF_GPU} python -u \
${INF_LAUNCH} ../drivers/run_ann_data_gen.py \
--training_dir $checkpoint_dir \
--init_model_dir $checkpoint_dir \
--model_type rdot_nll \
--output_dir ${output_dir} \
--cache_dir "${output_dir}/cache/" \
--srd_data_dir $preprocessed_data_dir \
--max_seq_length 512 \
--max_query_length $query_length \
--per_gpu_eval_batch_size 64
--topk_training 200
--negative_sample 20 \
--end_output_num 0
--inference \
"

echo $inf_cmd
eval $inf_cmd
