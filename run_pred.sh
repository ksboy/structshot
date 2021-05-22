#!/usr/bin/env bash

export ALGO=StructShot

export OUTPUT_DIR_NAME=output/pred/ccks/bin
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

export BERT_MODEL=/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
export CHECKPOINT=./output/role_crf/checkpoint-best/
# CUDA_VISIBLE_DEVICES=0 python3 -u run_pl_pred.py \
# CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:8888 --wait-for-client ./run_pl_pred.py \
CUDA_VISIBLE_DEVICES=0 python3 -u run_pl_pred.py \
--task_type EE \
--data_dir ./data/FewFC-main/rearranged/ \
--labels ./data/FewFC-main/event_schema/base.json \
--target_labels ./data/FewFC-main/event_schema/trans.json \
--train_fname base/train \
--sup_fname few/train \
--test_fname few/dev \
--model_name_or_path $BERT_MODEL \
--checkpoint $CHECKPOINT \
--output_dir $OUTPUT_DIR \
--algorithm $ALGO \
--train_batch_size 8 \
--eval_batch_size 2 \
--tau 0.05 \
--gpus 1 > $OUTPUT_DIR/output_$ALGO.log 2>&1 &


# export BERT_MODEL=/home/whou/workspace/pretrained_models/bert-base-uncased
# export CHECKPOINT=./output/conll-2003/checkpointepoch=2.ckpt
# CUDA_VISIBLE_DEVICES=0 python3 -u run_pl_pred.py \
# --task_type NER \
# --data_dir ./data/ \
# --labels ./data/conll-2003/labels.txt \
# --target_labels ./data/wnut/labels.txt \
# --train_fname conll-2003/train \
# --sup_fname support-wnut-5shot/0 \
# --test_fname wnut/test \
# --model_name_or_path $BERT_MODEL \
# --checkpoint $CHECKPOINT \
# --output_dir $OUTPUT_DIR \
# --algorithm $ALGO \
# --train_batch_size 8 \
# --eval_batch_size 2 \
# --tau 0.05 \
# --gpus 1 > $OUTPUT_DIR/output_$ALGO.log 2>&1 &