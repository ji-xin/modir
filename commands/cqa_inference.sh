#!/bin/bash

N_GPU=4  # 0 for using single gpu
INF_GPU="0,1,2,3"

if [[ $N_GPU = 0 ]]; then
  INF_LAUNCH=""
else
  INF_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_GPU} --master_port 11211"
fi

checkpoint_dir="../checkpoints/modir-cqadupstack-10k"

inference_subset() {
    TASK=${1}
    echo $TASK

    preprocessed_data_dir="../data/cqadupstack/${TASK}/preprocessed_data"
    output_dir="../inference_output/${TASK}"
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
    --per_gpu_eval_batch_size 64 \
    --topk_training 200 \
    --negative_sample 20 \
    --end_output_num 0 \
    --inference \
    "

    echo inf_cmd
    eval inf_cmd
}

for subset in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress; do
    inference_subset $subset
done

