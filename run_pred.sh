#!/usr/bin/env bash

export BERT_MODEL=/home/whou/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
export CHECKPOINT=./output/role_crf/checkpoint-best
export ALGO=StructShot

export OUTPUT_DIR_NAME=output/pred/ccks
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR


python3 -u run_pl_pred.py --data_dir ./data/FewFC-main/rearranged/trans/ \
--labels ./data/FewFC-main/event_schema/base.json \
--target_labels ./data/FewFC-main/event_schema/trans.json \
--train_fname train \
--sup_fname support \
--test_fname dev \
--model_name_or_path $BERT_MODEL \
--checkpoint $CHECKPOINT \
--output_dir $OUTPUT_DIR \
--algorithm $ALGO \
--train_batch_size 8 \
--eval_batch_size 2 \
--tau 0.05 \
--gpus 1 > $OUTPUT_DIR/output_$ALGO.log 2>&1 &

# python3 run_pl_pred.py --data_dir ../data/ \
# --labels ../data/labels.txt \
# --target_labels ../data/labels-wnut.txt \
# --train_fname train \
# --sup_fname support-wnut-5shot/0 \
# --test_fname test-wnut \
# --model_name_or_path $BERT_MODEL \
# --checkpoint $CHECKPOINT \
# --output_dir $OUTPUT_DIR \
# --algorithm $ALGO \
# --train_batch_size 8 \
# --eval_batch_size 2 \
# --tau 0.05 \
# --gpus 1 > $OUTPUT_DIR/output_$ALGO.log 2>&1 &
