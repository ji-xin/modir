dataset=${1}

raw_data_dir="${dataset}/marco-format"
preprocessed_data_dir="${dataset}/preprocessed_data"
mkdir -p $preprocessed_data_dir

max_query_length=64
if [[ $dataset = arguana ]]; then
  echo "arguana needs longer max_query_length"
  max_query_length=512
fi

python msmarco_data.py \
  --data_dir $raw_data_dir \
  --out_data_dir $preprocessed_data_dir \
  --model_type rdot_nll \
  --model_name_or_path roberta-base \
  --max_seq_length 512 \
  --max_query_length $max_query_length \
  --data_type 1 \
  --beir_dataset

rm $preprocessed_data_dir/*split*
python filter_train_qrel.py $preprocessed_data_dir/train-qrel.tsv

ln -s passages $preprocessed_data_dir/dev-passages
ln -s passages_meta $preprocessed_data_dir/dev-passages_meta
