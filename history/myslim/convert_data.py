# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from datasets import get_tfrecord


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'dataset_dir',
    '/home/mo/work/newNet/myslim/datasets/dataset3/train_test_validation',
    'The directory where the output TFRecords and temporary files are saved.')


tf.app.flags.DEFINE_string(
    'save_dir',
    '/home/mo/work/newNet/myslim/datasets/dataset3/tf_train_test_validation',
    'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_string(
    'split_name',
    'validation',
    'The directory where the output TFRecords and temporary files are saved.')

def main(_):

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    get_tfrecord.run(dataset_dir = FLAGS.dataset_dir,save_dir =  FLAGS.save_dir, split_name = None,all_data_to_one = False)

if __name__ == '__main__':
  tf.app.run()
