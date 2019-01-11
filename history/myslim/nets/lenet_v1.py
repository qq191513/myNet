from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
end_points = collections.OrderedDict()

slim = tf.contrib.slim


def lenet_v1(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet_v1'):



  with tf.variable_scope(scope, 'LeNet_v1', [images]):
    net = end_points['conv1'] = slim.conv2d(images, 32, [1, 1], scope='conv1')
    net = end_points['conv2'] = slim.conv2d(images, 32, [3, 3], scope='conv2')
    net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = end_points['conv3'] = slim.conv2d(images, 32, [1, 1], scope='conv3')
    net = end_points['conv4'] = slim.conv2d(net, 64, [3, 3], scope='conv4')
    net = end_points['pool4'] = slim.max_pool2d(net, [2, 2], 2, scope='pool4')


    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = end_points['fc5'] = slim.fully_connected(net, 1024, scope='fc5')
    if not num_classes:
      return net, end_points
    net = end_points['dropout5'] = slim.dropout(
        net, dropout_keep_prob, is_training=is_training, scope='dropout5')
    logits = end_points['Logits'] = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc6')

  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
lenet_v1.default_image_size = 28


def lenet_v1_arg_scope(weight_decay=0.0):
  """Defines the default lenet argument scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.

  Returns:
    An `arg_scope` to use for the inception v3 model.
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
      activation_fn=tf.nn.relu) as sc:
    return sc
