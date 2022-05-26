raw_data_dir="msmarco/raw_data/"
preprocessed_data_dir="msmarco/preprocessed_data"
mkdir -p $preprocessed_data_dir


python msmarco_data.py \
  --data_dir $raw_data_dir \
  --out_data_dir $preprocessed_data_dir \
  --model_type rdot_nll \
  --model_name_or_path roberta-base \
  --max_seq_length 512 \
  --data_type 1

rm $preprocessed_data_dir/*split*