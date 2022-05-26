# this only works on the debugging server

# blob_checkpoint="pretrained-checkpoints/ance-public"
blob_checkpoint="da-checkpoints/cqa-all/warmupFP-confusion-lamb5e-1-shrinkHalf10k-lr5e-6-dclr3e-5-nowarm-ckpt2500"

seq_length=512
data_type=1
model_type=rdot_nll
pretrained_checkpoint_dir="../pretrained_checkpoint"
rm -rf ${pretrained_checkpoint_dir}
ln -sf "${ROOT}/rblob/${blob_checkpoint}" $pretrained_checkpoint_dir
model_dir="../saved_model"
model_ann_data_dir="${model_dir}/ann_data_inf"
mkdir -p $model_ann_data_dir

N_TRAIN_GPU=2
TRAIN_GPU="0,1"
if [[ $N_TRAIN_GPU = 0 ]]; then
  TRAIN_LAUNCH=""
else
  TRAIN_LAUNCH="-m torch.distributed.launch --nproc_per_node=${N_TRAIN_GPU} --master_port 11211"
fi

preprocessed_data_dir="../preprocessed_data"

initial_data_gen_cmd="\
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python -u \
${TRAIN_LAUNCH} ../drivers/run_ann_data_gen.py \
--training_dir $pretrained_checkpoint_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type \
--output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}/cache/" \
--srd_data_dir $preprocessed_data_dir \
--max_seq_length $seq_length \
--per_gpu_eval_batch_size 64 --topk_training 200 --negative_sample 20 \
--end_output_num 0 --inference \
"


inference_subset() {
    TASK=${1}
    echo $TASK
    blob_data="${ROOT}/rblob/cqa-all/subsets/${TASK}/preprocessed_data/rdot_nll-512-1"
    rm -rf ${preprocessed_data_dir}
    ln -sf ${blob_data} ${preprocessed_data_dir}
    rm -f ${model_ann_data_dir}/*.pb

    echo $initial_data_gen_cmd
    eval $initial_data_gen_cmd

    ls -lah ${model_ann_data_dir}

    mkdir -p ${ROOT}/wblob/${blob_checkpoint}/cqa-all
    cp -r ${model_ann_data_dir} ${ROOT}/wblob/${blob_checkpoint}/cqa-all/${TASK}-ann_data_inf
}

for subset in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress; do
    inference_subset $subset
done

