from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import collections
end_points = collections.OrderedDict()

slim = tf.contrib.slim

@slim.add_arg_scope
def _fire_module(inputs,
                 squeeze_depth,
                 outputs_collections=None,
                 scope=None,
                 use_bypass=False
                 ):
  """
  Creates a fire module

  Arguments:
      x                 : input
      squeeze_depth     : number of filters of squeeze. The filtersize of expand is 4 times of squeeze
      use_bypass        : if True then a bypass will be added
      name              : name of module e.g. fire123

  Returns:
      x                 : returns a fire module
  """

  with tf.variable_scope(scope, 'fire', [inputs]) as sc:
    with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
      expand_depth = squeeze_depth * 4
      # squeeze
      squeeze = end_points[scope+ '/squeeze_1X1'] = slim.conv2d(inputs, squeeze_depth, [1, 1], scope="squeeze_1X1")

      # expand
      expand_1x1 = end_points[scope+ '/expand_1x1'] = slim.conv2d(squeeze, expand_depth, [1, 1], scope="expand_1x1")
      expand_3x3 = end_points[scope+ '/expand_3x3'] = slim.conv2d(squeeze, expand_depth, [3, 3], scope="expand_3x3")

      # concat
      x_ret = tf.concat([expand_1x1, expand_3x3], axis=3)

      # fire 3/5/7/9
      if use_bypass:
        x_ret = x_ret + inputs
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)



def lenet_v2(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet_v2'):

  # end_points = {}
  compression = 1.0
  with tf.variable_scope(scope, 'LeNet_v2', [images]):
    net = end_points['conv1'] = slim.conv2d(images, 32, [3, 3], scope='conv1')
    net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = end_points['fire2']= _fire_module(net, int(16 * compression), scope="fire2")

    net = end_points['conv3'] = slim.conv2d(net, 128, [3, 3], scope='conv3')
    net = end_points['pool3'] = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = end_points['fire4']= _fire_module(net, int(32 * compression), scope="fire4")
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
lenet_v2.default_image_size = 28


def lenet_v2_arg_scope(weight_decay=0.0):
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


import numpy as np
if __name__ == "__main__":
    # inputs = tf.random_normal([1, 32, 32, 3])
    inputs = tf.random_normal([1, lenet_v2.default_image_size, lenet_v2.default_image_size, 3])
    with slim.arg_scope(lenet_v2_arg_scope()):
         logits, end_points= lenet_v2(inputs,10)

    #########  parament numbers###########
    from functools import reduce
    from operator import mul
    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx parament numbers is : %d' % get_num_params())
    #######################################

    # writer = tf.summary.FileWriter("./alexnet", graph=tf.get_default_graph())
    print("Layers")
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    print("Parameters")
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(end_points['Predictions'])
#         print(pred)
        print(np.argmax(pred,1))
#         print(pred[:,np.argmax(pred,1)])


