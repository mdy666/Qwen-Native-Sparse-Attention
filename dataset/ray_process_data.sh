set -f
set -x
INPUT="/sharedata/mdy/data/fineweb*/data/*parquet"
OUTPUT=/sharedata/mdy/data/pretrain/tmp3/fineweb_edu_zh
TOKENIZER=/sharedata/mdy/models/SauerkrautLM-Mixtral-8x7B-Instruct
WORKERS=80

ray stop
ray start --head

python tools/ray_process_data.py \
    --input $INPUT \
    --output-prefix $OUTPUT \
    --json-keys text \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER \
    --workers $WORKERS \
    --append-eod \
    --partitions 8

ray stop


# python setup.py build_ext --inplace