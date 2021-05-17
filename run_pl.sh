#!/usr/bin/env bash

export MAX_LENGTH=256
export BERT_MODEL=/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
# export BERT_MODEL=/home/whou/workspace/pretrained_models/bert-base-uncased
export BATCH_SIZE=16
export NUM_EPOCHS=25
export SEED=1

export OUTPUT_DIR_NAME=output/role
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# # python3 run_pl_ner.py \
# # CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_pl_ner.py \
# python3 run_pl_ner.py \
# --task_type EE \
# --data_dir ./data/FewFC-main/rearranged/base/ \
# --labels ./data/FewFC-main/event_schema/base.json \
# --model_name_or_path $BERT_MODEL \
# --output_dir $OUTPUT_DIR \
# --max_seq_length  $MAX_LENGTH \
# --num_train_epochs $NUM_EPOCHS \
# --train_batch_size $BATCH_SIZE \
# --seed $SEED \
# --gpus 1 \
# --do_train \
#  > $OUTPUT_DIR/output.log 2>&1 &


# python3 run_pl_ner.py \
# CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_pl_ner.py \
python3 run_pl_ner.py \
--task_type EE \
--data_dir ./data/FewFC-main/rearranged/base/ \
--labels ./data/FewFC-main/event_schema/base.json \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--gpus 1 \
--do_train \
 > $OUTPUT_DIR/output.log 2>&1 &
