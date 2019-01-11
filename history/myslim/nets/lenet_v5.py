from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import collections
end_points = collections.OrderedDict()

slim = tf.contrib.slim

def mix_module(net):
    end_point = 'Mixed_out1_inception_1'

    with tf.variable_scope(end_point):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_3'):
                branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
            net = tf.concat(
                axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
    return net


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
    # hhhh = slim.utils.convert_collection_to_dict('hhdfgh')
    # x_ret_v1 = slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)
    return slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)


def lenet_v5(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='lenet_v5'):

  # end_points = {}
  compression = 1.0
  with tf.variable_scope(scope, 'lenet_v5', [images]):
    net = end_points['conv1'] = slim.conv2d(images, 32, [3, 3], scope='conv1')
    net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = end_points['fire2']= _fire_module(net, int(16 * compression), scope="fire2")
    net = end_points['conv3'] = slim.conv2d(net, 144, [3, 3], scope='conv3')
    # net = end_points['pool3'] = slim.max_pool2d(net, [2, 2], 2, scope='pool3')
    net = end_points['fire4']= _fire_module(net, int(32 * compression), scope="fire4")
    out1= mix_module(net)
    out1 = slim.max_pool2d(out1, [2, 2], 2, scope='out1_pool')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool3')

    #全部压扁
    out1 = slim.flatten(out1)
    out2 = slim.flatten(net)
    end_points['Flatten'] = net

    # 各自的全连接
    out1 = end_points['fc5'] = slim.fully_connected(out1, 512, scope='fc5')
    out2 = end_points['fc6'] = slim.fully_connected(out2, 512, scope='fc6')

    # 各自得预测
    #out1
    out1 = end_points['dropout5'] = slim.dropout(
        out1, dropout_keep_prob, is_training=is_training, scope='dropout5')
    Logits_out1 = end_points['Logits_out1'] = slim.fully_connected(
        out1, num_classes, activation_fn=None, scope='fc7')

    # out2
    out2 = end_points['dropout6'] = slim.dropout(
        out2, dropout_keep_prob, is_training=is_training, scope='dropout6')
    Logits_out2 = end_points['Logits_out2'] = slim.fully_connected(
        out2, num_classes, activation_fn=None, scope='fc8')

    end_points['Predictions_1'] = prediction_fn(Logits_out1, scope='Predictions')
    end_points['Predictions_2'] = prediction_fn(Logits_out2, scope='Predictions')

    return Logits_out1,Logits_out2,end_points

lenet_v5.default_image_size = 28


def lenet_v5_arg_scope(weight_decay=0.0):
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
    inputs = tf.random_normal([1, lenet_v5.default_image_size, lenet_v5.default_image_size, 3])
    with slim.arg_scope(lenet_v5_arg_scope()):
        Logits_out1, Logits_out2, end_points= lenet_v5(inputs,10)

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
        pred_1 = sess.run(end_points['Predictions_1'])
        pred_2 = sess.run(end_points['Predictions_2'])
        Logits_out1_1 = sess.run(Logits_out1)
#         print(pred)

        print(np.argmax(pred_1,1))
        print(np.argmax(pred_2, 1))
        print(Logits_out1_1)
#         print(pred[:,np.argmax(pred,1)])


