"""A script for matrix capsule with EM routing, settings for train/tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import capsules
from dataset.load_data_cifar10_32x32 import get_data_set

import tensorflow as tf

slim = tf.contrib.slim



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'data_dir', '/home/mo/work/data_set/', 'Data Directory'
)
tf.app.flags.DEFINE_string(
  'train_dir', 'log/train/', 'Train Directory.'
)
tf.app.flags.DEFINE_string(
  'tests_dir', 'log/tests/', 'Tests Directory.'
)
tf.app.flags.DEFINE_string(
  'checkpoint_path', FLAGS.train_dir,
  'The directory where the model was written to or an absolute path to a checkpoint file.'
)

tf.app.flags.DEFINE_integer(
  'batch_size', 24, 'Train/Tests Batch Size.'
)
tf.app.flags.DEFINE_boolean(
  'is_training', True, 'Train/Tests'
)
