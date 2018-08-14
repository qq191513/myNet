#!/bin/bash

set -e

###############dataset1
TRAIN_DIR=log/inception_v1/cifar10_process_cifarnet
DATASET_DIR=datasets/cifar-10
DATASET_NAME=cifar10
MODEL_NAME=inception_v1
PREPROCESSING_NAME=cifarnet
################################


python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --model_name=${MODEL_NAME} \
  --max_number_of_steps=10000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \