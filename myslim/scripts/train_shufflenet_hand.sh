#!/bin/bash

set -e

MODEL_NAME=shufflenet
PREPROCESSING_NAME=lenet

###############dataset1
TRAIN_DIR=log/train_log_shufflenet
DATASET_DIR=datasets/dataset1/tf_train_test_validation/
DATASET_NAME=italy_dataset1
################################

################dataset2
#TRAIN_DIR=log/train_log_letnet_v1_dataset2
#DATASET_DIR=datasets/dataset2/tf_train_test_validation/
#DATASET_NAME=italy_dataset2
################################

###############dataset3
#TRAIN_DIR=log/train_log_letnet_v1_dataset3
#DATASET_DIR=datasets/dataset3/tf_train_test_validation/
#DATASET_NAME=italy_dataset3
################################

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --save_interval_secs=300 \
  --max_number_of_steps=10000 \
  --batch_size=50 \
  --learning_rate=0.01 \
  --save_summaries_secs=60 \
  --log_every_n_steps=50 \
  --optimizer=adam \
  --learning_rate_decay_type=exponential \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --batch_size=100 \
  --checkpoint_path=${TRAIN_DIR} \
  --preprocessing_name=${PREPROCESSING_NAME} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
