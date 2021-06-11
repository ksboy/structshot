#!/usr/bin/env bash

export ALGO=NNShot

export OUTPUT_DIR_NAME=output/pred/ccks/trigger/entity/$ALGO/2
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

# export BERT_MODEL=/hy-nas/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
# export CHECKPOINT=/hy-nas/workspace/pretrained_models/chinese_bert_wwm_ext_pytorch/
export BERT_MODEL=/hy-nas/workspace/code_repo/ner/output/ccks/trigger/base/joint/checkpoint-best/
export CHECKPOINT=/hy-nas/workspace/code_repo/ner/output/ccks/trigger/base/joint/checkpoint-best/
# CUDA_VISIBLE_DEVICES=0 python3 -m debugpy --listen 0.0.0.0:9999 --wait-for-client ./run_pl_pred_entity.py \
CUDA_VISIBLE_DEVICES=0 python3 run_pl_pred_entity.py \
--task_type EE \
--sub_task trigger \
--data_dir ./data/FewFC-main/rearranged/ \
--labels ./data/FewFC-main/event_schema/base.json \
--target_labels ./data/FewFC-main/event_schema/trans.json \
--train_fname few/train \
--sup_fname few/trigger/train \
--test_fname few/trigger/identification \
--model_name_or_path $BERT_MODEL \
--checkpoint $CHECKPOINT \
--output_dir $OUTPUT_DIR \
--algorithm $ALGO \
--train_batch_size 8 \
--eval_batch_size 64 \
--tau 0.05 \
--gpus 1  > $OUTPUT_DIR/output_$ALGO.log 2>&1 &


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