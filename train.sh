set -f
DATA_PATH="/sharedata/mdy/data/pretrain/mixtral/fineweb_edu_en_text_document"
TOKENIZER="/sharedata/mdy/models/SauerkrautLM-Mixtral-8x7B-Instruct"
MODEl_CONFIG="./qwen2/config7B.json"
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=256
OUTPUT_DIR="/sharedata/mdy/models/base-7b-ckpt"
TENSORBOARD="./log/7B/base-fp8"
MAX_SEQ_LEN=4096
MAX_STEPS=5000

N_GPUS=8
torchrun --nproc_per_node=$N_GPUS train.py $@ \
    --model-config $MODEl_CONFIG \
    --tokenizer $TOKENIZER \
    --data-path $DATA_PATH \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --output_dir $OUTPUT_DIR \
    --log_dir $TENSORBOARD \
    --max_seq_len $MAX_SEQ_LEN \
    --max_steps $MAX_STEPS

CMD = '''
# base-bf16
bash train.sh --deepspeed
# base-fp8
bash train.sh --deepspeed --fp8 --fp8-pattern proj
# nsa-bf16
bash train.sh --deepspeed --nsa
# nsa-fp8
bash train.sh --deepspeed --nsa --fp8 --fp8-pattern proj
'''


# python train.py $@ \
#     --model_path $MODEl_PATH \
#     --data_paths $DATA_PATHS \
#     --micro_batch_size $MICRO_BATCH_SIZE \
#     --global_batch_size $GLOBAL_BATCH_SIZE \
#     --output_dir $OUTPUT_DIR \
#     --max_seq_len $MAX_SEQ_LEN \
#     --max_steps $MAX_STEPS
