#!/usr/bin/env bash

export MAX_LENGTH=128
export BERT_MODEL=/home/whou/workspace/pretrained_models/bert-base-uncased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=1

export OUTPUT_DIR_NAME=output
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# python3 run_pl_ner.py \
CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_pl_ner.py \
--data_dir ../data/ \
--labels ../data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--gpus 1 \
--do_train \
--do_predict # > $OUTPUT_DIR/output.log 2>&1 &
