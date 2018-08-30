# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer
import numpy as np
from src.model.squeezenet import squeezenet
from src.model.squeezenet import squeezenet_arg_scope
from collections import namedtuple
slim = tf.contrib.slim
import time
import threading
min_depth =16
depth_multiplier=1.0
concat_dim = 3
depth = lambda d: max(int(d * depth_multiplier), min_depth)
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
#Numpy花式索引，获取所有元素的出现次数
def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def find_dict_max_key(result = {}):
    max_value = 0
    max_key = 0
    for key in result.keys():
        if result[key] > max_value:
            max_value = result[key]
            max_key = key
    return max_key

def print_and_save_txt(str=None,filename=r'log.txt'):
    with open(filename, "a+") as log_writter:
        print(str)
        log_writter.write(str)

DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
_CONV_DEFS = [
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
]

def _fixed_padding(inputs, kernel_size, rate=1):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.

  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                           kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
  pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
  pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
  pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                  [pad_beg[1], pad_end[1]], [0, 0]])
  return padded_inputs


class ConvNet():
    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=44,is_training =True,debug_mode = False):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers
        self.is_training  = is_training
        self.debug_mode =debug_mode


        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, self.image_size, self.image_size, self.n_channel],
            name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')

        # 网络输出
        conv_layer1 = ConvLayer(
            input_shape=(None, image_size, image_size, n_channel), n_size=3, n_filter=3,
            stride=1,activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv1')

        dense_layer_1 = DenseLayer(
            input_shape=(None, 512),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_1')

        dense_layer_2 = DenseLayer(
            input_shape=(None, 490),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_2')

        dense_layer_3 = DenseLayer(
            input_shape=(None, 200),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_3')


        # 数据流
        #父层
        basic_conv = conv_layer1.get_output(input=self.images)

        #子层1
        hidden_conv_1 = self.residual_inference(images=basic_conv, scope_name='son_1_1')
        hidden_conv_1 = self.son_google_v2_part(hidden_conv_1,'son_1_2')
        input_dense_1 = tf.reduce_mean(hidden_conv_1, reduction_indices=[1, 2])
        self.logits_1 = dense_layer_1.get_output(input=input_dense_1)

        #子层1--孙1 densenet
        self.logits_1_1 = self.grand_son_1_1(hidden_conv_1,scope_name='grand_son_1_1')

        #子层1--孙2 densenet_bc
        self.logits_1_2 = self.grand_son_1_2(hidden_conv_1,scope_name='grand_son_1_2')

        # 子层1--孙3 mobilenet --- depth conv L3
        self.logits_1_3 = self.grand_son_1_3(hidden_conv_1, scope_name='grand_son_1_3')

        #子层2
        hidden_conv_2 = self.residual_inference(images=basic_conv, scope_name='son_2_1')
        hidden_conv_2 = self.son_google_v3_part_2(hidden_conv_2,'son_2_2')
        input_dense_2 = tf.reduce_mean(hidden_conv_2, reduction_indices=[1, 2])
        self.logits_2 = dense_layer_2.get_output(input=input_dense_2)


        # 子层2--孙1   -- depth conv L8
        self.logits_2_1 = self.grand_son_2_1(hidden_conv_2, scope_name='grand_son_2_1')

        # 子层2--孙2   -- ddensenet_bc
        self.logits_2_2 = self.grand_son_2_2(hidden_conv_2, scope_name='grand_son_2_2')

        # 子层2--孙3   -- Fire Model
        self.logits_2_3 = self.grand_son_2_3(hidden_conv_2, scope_name='grand_son_2_3')


        # 子层3
        hidden_conv_3 = self.residual_inference(images=basic_conv, scope_name='son_3')
        hidden_conv_3 =self.son_cnn_part_3(hidden_conv_3,scope_name='net_3_2')
        input_dense_3= tf.reduce_mean(hidden_conv_3, reduction_indices=[1, 2])
        self.logits_3 = dense_layer_3.get_output(input=input_dense_3)

        # 目标函数
        logits_list=[self.logits_1,self.logits_1_1,self.logits_1_2,self.logits_1_3,
                    self.logits_2,self.logits_2_1,self.logits_2_2,self.logits_2_3,
                    self.logits_3
                    ]

        self.objective = 0
        for item in logits_list:
            self.objective_item = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=item, labels=self.labels))
            self.objective += self.objective_item


        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))

        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 75000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                                     lambda: tf.constant(0.001),
                                                     lambda: tf.constant(0.001))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        #son1
        correct_prediction_1 = tf.equal(self.labels, tf.argmax(self.logits_1, 1))
        self.accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, 'float'))

        # 观察值son1的  grand son1
        correct_prediction_1_1 = tf.equal(self.labels, tf.argmax(self.logits_1_1, 1))
        self.accuracy_1_1 = tf.reduce_mean(tf.cast(correct_prediction_1_1, 'float'))

        # 观察值son1的  grand son2
        correct_prediction_1_2 = tf.equal(self.labels, tf.argmax(self.logits_1_2, 1))
        self.accuracy_1_2 = tf.reduce_mean(tf.cast(correct_prediction_1_2, 'float'))

        # 观察值son1的 grand son3
        correct_prediction_1_3 = tf.equal(self.labels, tf.argmax(self.logits_1_3, 1))
        self.accuracy_1_3 = tf.reduce_mean(tf.cast(correct_prediction_1_3, 'float'))

        # son2
        correct_prediction_2 = tf.equal(self.labels, tf.argmax(self.logits_2, 1))
        self.accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, 'float'))

        # 观察值son2的 grand son1
        correct_prediction_2_1 = tf.equal(self.labels, tf.argmax(self.logits_2_1, 1))
        self.accuracy_2_1 = tf.reduce_mean(tf.cast(correct_prediction_2_1, 'float'))

        # 观察值son2的 grand son2
        correct_prediction_2_2 = tf.equal(self.labels, tf.argmax(self.logits_2_2, 1))
        self.accuracy_2_2 = tf.reduce_mean(tf.cast(correct_prediction_2_2, 'float'))

        # 观察值son2的 grand son3
        correct_prediction_2_3 = tf.equal(self.labels, tf.argmax(self.logits_2_3, 1))
        self.accuracy_2_3 = tf.reduce_mean(tf.cast(correct_prediction_2_3, 'float'))

        # son3
        correct_prediction_3 = tf.equal(self.labels, tf.argmax(self.logits_3, 1))
        self.accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, 'float'))


    def grand_son_1_1(self,hidden_conv_1,scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):
                    bc_mode = False
                    dropout_keep_prob = 0.8
                    with tf.variable_scope("block_3"):
                        hidden_conv_1_1 = slim.repeat(hidden_conv_1, 4, self.add_internal_layer,
                                                      300, self.is_training, bc_mode, dropout_keep_prob)
                        train = self.is_training
                        train = True
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_1, self.n_classes, train)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits


    def grand_son_1_2(self, hidden_conv_1_2, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):
                    bc_mode =True
                    dropout_keep_prob = 0.8
                    with tf.variable_scope("block_3"):
                        hidden_conv_1_2 = slim.repeat(hidden_conv_1_2, 4, self.add_internal_layer,
                                                      300, self.is_training,  True, 0.8)
                        hidden_conv_1_2 = self.transition_layer(hidden_conv_1_2, 500, self.is_training)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_2, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def grand_son_arg_scope_1_2(self,is_training=True,
                                weight_decay=0.00004,
                                stddev=0.09,
                                regularize_depthwise=False,
                                batch_norm_decay=0.9997,
                                batch_norm_epsilon=0.001):
        """Defines the default MobilenetV1 arg scope.

        Args:
          is_training: Whether or not we're training the model.
          weight_decay: The weight decay to use for regularizing the model.
          stddev: The standard deviation of the trunctated normal weight initializer.
          regularize_depthwise: Whether or not apply regularization on depthwise.
          batch_norm_decay: Decay for batch norm moving average.
          batch_norm_epsilon: Small float added to variance to avoid dividing by zero
            in batch norm.

        Returns:
          An `arg_scope` to use for the mobilenet v1 model.
        """
        batch_norm_params = {
            'is_training': is_training,
            'center': True,
            'scale': True,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
        }

        # Set weight_decay for weights in Conv and DepthSepConv layers.
        weights_init = tf.truncated_normal_initializer(stddev=stddev)
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        if regularize_depthwise:
            depthwise_regularizer = regularizer
        else:
            depthwise_regularizer = None
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_initializer=weights_init,
                            activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                    with slim.arg_scope([slim.separable_conv2d],
                                        weights_regularizer=depthwise_regularizer) as sc:
                        return sc

    def grand_son_1_3(self, net, scope_name):
        global_pool = True
        padding = 'SAME'
        use_explicit_padding = True
        spatial_squeeze = True
        dropout_keep_prob = 0.9
        final_endpoint = 'Conv2d_3'
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.grand_son_arg_scope_1_2(is_training= self.is_training)):
                if use_explicit_padding:
                    padding = 'VALID'
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
                    conv_defs = _CONV_DEFS
                    output_stride=8
                    current_stride = 1
                    rate = 1

                    for i, conv_def in enumerate(conv_defs):
                        end_point_base = 'Conv2d_%d' % i
                        end_point = end_point_base
                        if output_stride is not None and current_stride == output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= conv_def.stride
                        else:
                            layer_stride = conv_def.stride
                            layer_rate = 1
                            current_stride *= conv_def.stride



                        if use_explicit_padding:
                            net = _fixed_padding(net, conv_def.kernel, layer_rate)
                        net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                    depth_multiplier=1,
                                                    stride=layer_stride,
                                                    rate=layer_rate,
                                                    normalizer_fn=slim.batch_norm,
                                                    scope=end_point)
                        if end_point == final_endpoint:
                            break

                    with tf.variable_scope('Logits'):
                        if global_pool:
                            # Global average pooling.
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')


                        # 1 x 1 x 512
                        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                        logits = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, scope='Conv2d_1c_1x1')
                        if spatial_squeeze:
                            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        return logits
    def grand_son_2_1(self, net, scope_name):
        global_pool = True
        padding = 'SAME'
        use_explicit_padding = True
        spatial_squeeze = True
        dropout_keep_prob = 0.9
        final_endpoint = 'Conv2d_7'
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.grand_son_arg_scope_1_2(is_training= self.is_training)):
                if use_explicit_padding:
                    padding = 'VALID'
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
                    conv_defs = _CONV_DEFS
                    output_stride=8
                    current_stride = 1
                    rate = 1

                    for i, conv_def in enumerate(conv_defs):
                        end_point_base = 'Conv2d_%d' % i
                        end_point = end_point_base
                        if output_stride is not None and current_stride == output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= conv_def.stride
                        else:
                            layer_stride = conv_def.stride
                            layer_rate = 1
                            current_stride *= conv_def.stride



                        if use_explicit_padding:
                            net = _fixed_padding(net, conv_def.kernel, layer_rate)
                        net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                    depth_multiplier=1,
                                                    stride=layer_stride,
                                                    rate=layer_rate,
                                                    normalizer_fn=slim.batch_norm,
                                                    scope=end_point)
                        if end_point == final_endpoint:
                            break
                    with tf.variable_scope('Logits'):
                        if global_pool:
                            # Global average pooling.
                            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')


                        # 1 x 1 x 512
                        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                        logits = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                                             normalizer_fn=None, scope='Conv2d_1c_1x1')
                        if spatial_squeeze:
                            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        return logits
    def grand_son_2_2(self, hidden_conv_2_2, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):
                    bc_mode =True
                    dropout_keep_prob = 0.8
                    with tf.variable_scope("block_3"):
                        hidden_conv_2_2 = slim.repeat(hidden_conv_2_2, 3, self.add_internal_layer,
                                                      400, self.is_training,  True, 0.9)
                        hidden_conv_2_2 = self.transition_layer(hidden_conv_2_2, 600,
                                                                self.is_training, dropout_keep_prob,0.9)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_2_2, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    @slim.add_arg_scope
    def _fire_module(self,inputs,
                     squeeze_depth,
                     outputs_collections=None,
                     scope=None,
                     use_bypass=False
                     ):
        with tf.variable_scope(scope, 'fire', [inputs]) as sc:
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                expand_depth = squeeze_depth * 4
                # squeeze
                squeeze = slim.conv2d(inputs, squeeze_depth, [1, 1], scope="squeeze_1X1")

                # expand
                expand_1x1 = slim.conv2d(squeeze, expand_depth, [1, 1], scope="expand_1x1")
                expand_3x3 = slim.conv2d(squeeze, expand_depth, [3, 3], scope="expand_3x3")

                # concat
                x_ret = tf.concat([expand_1x1, expand_3x3], axis=3)

                # fire 3/5/7/9
                if use_bypass:
                    x_ret = x_ret + inputs
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)

    def grand_son_2_3(self, hidden_conv_2_3, scope_name):
        compression = 1.0
        use_bypass = False
        dropout_keep_prob = 0.9
        global_pool =True
        spatial_squeeze =True
        with tf.variable_scope(scope_name) as sc:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d, self._fire_module]):
                    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
                        # fire5
                        net = self._fire_module(hidden_conv_2_3, int(32 * compression), scope="fire5", use_bypass=use_bypass)
                        # fire6
                        net = self._fire_module(net, int(48 * compression), scope="fire6")

                        # fire7
                        net = self._fire_module(net, int(48 * compression), scope="fire7", use_bypass=use_bypass)

                        # fire8
                        net = self._fire_module(net, int(64 * compression), scope="fire8")

                        if dropout_keep_prob:
                            net = slim.dropout(net, keep_prob=dropout_keep_prob, scope="dropout")
                        ####################################
                        # conv10
                        # net = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                        #                   normalizer_fn=None, scope='conv10')
                        #
                        # with tf.variable_scope('Logits'):
                        #     # avgpool10
                        #     if global_pool:
                        #         # Global average pooling.
                        #         net = tf.reduce_mean(net, [1, 2], name='pool10', keep_dims=True)
                        #     # squeeze the axis
                        #     if spatial_squeeze:
                        #         logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        ##############################
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(net, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
                        ###########################

            return logits

    def son_cnn_part_3(self, hidden_conv, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                hidden_conv = slim.conv2d(hidden_conv, depth(200), [3, 3], scope='Conv2d_1')
        return hidden_conv

    def son_google_v3_part_2(self, hidden_conv, scope_name):
        # google_v2_part InceptionV3
        with tf.variable_scope(scope_name):
            with slim.arg_scope(
                    [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1,
                    padding='SAME'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(hidden_conv, depth(80), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(hidden_conv, depth(90), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, depth(90), [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, depth(90), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(hidden_conv, depth(120), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, depth(90), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, depth(90), [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, depth(90), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(hidden_conv, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(50), [1, 1], scope='Conv2d_0b_1x1')
                    hidden_conv = tf.concat(
                        axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        return hidden_conv

    def son_google_v2_part(self,hidden_conv,scope_name):

        # google_v2_part
        with tf.variable_scope(scope_name):
            with slim.arg_scope(
                    [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1,
                    padding='SAME'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(hidden_conv, depth(176), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        hidden_conv, depth(96), [1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                           scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        hidden_conv, depth(80), [1, 1],
                        weights_initializer=trunc_normal(0.09),
                        scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(112), [3, 3],
                                           scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(112), [3, 3],
                                           scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(hidden_conv, [2, 2], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, depth(64), [1, 1],
                        weights_initializer=trunc_normal(0.1),
                        scope='Conv2d_0b_1x1')
                hidden_conv = tf.concat(
                    axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])


        return hidden_conv

    def transition_layer(self,_input, num_filter, training=True, dropout_keep_prob=0.8, reduction=1.0):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        num_filter = int(num_filter * reduction)
        _output = self.composite_function(_input, num_filter, training, kernel_size=[1, 1])
        if training:
            _output = slim.dropout(_output, dropout_keep_prob)
        _output = slim.avg_pool2d(_output, [2, 2])
        return _output

    def trainsition_layer_to_classes(self,_input, n_classes=10, training=True):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        _output = slim.batch_norm(_input, is_training=training)
        # L.scale in caffe
        _output = tf.nn.relu(_output)
        last_pool_kernel = int(_output.get_shape()[-2])
        _output = slim.avg_pool2d(_output, [last_pool_kernel, last_pool_kernel])
        logits = slim.fully_connected(_output, n_classes)
        return logits

    def composite_function(self,_input, out_features, training=True, dropout_keep_prob=0.8, kernel_size=[3, 3]):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = slim.batch_norm(_input, is_training=training)  # !!need op
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = slim.conv2d(output, out_features, kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            if training:
                output = slim.dropout(output, dropout_keep_prob)
        return output

    def bottleneck(self,_input, out_features, training=True, dropout_keep_prob=0.8):
        with tf.variable_scope("bottleneck"):
            inter_features = out_features * 4
            output = slim.batch_norm(_input, is_training=training)  # !!need op
            output = tf.nn.relu(output)
            output = slim.conv2d(_input, inter_features, [1, 1], padding='VALID')
            if training:
                output = slim.dropout(output, dropout_keep_prob)
        return output

    def add_internal_layer(self,_input, growth_rate, training=True, bc_mode=False, dropout_keep_prob=1.0,
                           scope="inner_layer"):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        with tf.variable_scope(scope):
            if not bc_mode:
                _output = self.composite_function(_input, growth_rate, training)
                if training:
                    _output = slim.dropout(_output, dropout_keep_prob)

            elif bc_mode:
                bottleneck_out = self.bottleneck(_input, growth_rate, training)
                _output = self.composite_function(bottleneck_out, growth_rate, training)
                if training:
                    _output = slim.dropout(_output, dropout_keep_prob)

            # concatenate _input with out from composite function
            # the only diffenence between resnet and densenet
            output = tf.concat(axis=3, values=(_input, _output))
            return output

    def transition_layer(self,_input, num_filter, training=True, dropout_keep_prob=0.8, reduction=1.0):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        num_filter = int(num_filter * reduction)
        _output = self.composite_function(_input, num_filter, training, kernel_size=[1, 1])
        if training:
            _output = slim.dropout(_output, dropout_keep_prob)
        _output = slim.avg_pool2d(_output, [2, 2])
        return _output

    def residual_inference(self, images,scope_name):
        with tf.variable_scope(scope_name):
            n_layers = int((self.n_layers - 2) / 6)
            # 网络结构
            conv_layer0_list = []
            conv_layer0_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 3),
                    n_size=3, n_filter=64, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv0'))

            conv_layer1_list = []
            for i in range(1, n_layers+1):
                conv_layer1_list.append(
                    ConvLayer(
                        input_shape=(None, self.image_size, self.image_size, 64),
                        n_size=3, n_filter=64, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2*i-1)))
                conv_layer1_list.append(
                    ConvLayer(
                        input_shape=(None, self.image_size, self.image_size, 64),
                        n_size=3, n_filter=64, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2*i)))

            conv_layer2_list = []
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 64),
                    n_size=3, n_filter=128, stride=2, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv2_1'))
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size)/2, int(self.image_size)/2, 128),
                    n_size=3, n_filter=128, stride=1, activation='none',
                    batch_normal=True, weight_decay=1e-4, name='conv2_2'))
            for i in range(2, n_layers+1):
                conv_layer2_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size/2), int(self.image_size/2), 128),
                        n_size=3, n_filter=128, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2*i-1)))
                conv_layer2_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size/2), int(self.image_size/2), 128),
                        n_size=3, n_filter=128, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2*i)))

            conv_layer3_list = []
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/2), int(self.image_size/2), 128),
                    n_size=3, n_filter=256, stride=2, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv3_1'))
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size/4), int(self.image_size/4), 256),
                    n_size=3, n_filter=256, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv3_2'))
            for i in range(2, n_layers+1):
                conv_layer3_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size/4), int(self.image_size/4), 256),
                        n_size=3, n_filter=256, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2*i-1)))
                conv_layer3_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size/4), int(self.image_size/4), 256),
                        n_size=3, n_filter=256, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2*i)))


            # 数据流
            hidden_conv = conv_layer0_list[0].get_output(input=images)

            for i in range(0, n_layers):
                hidden_conv1 = conv_layer1_list[2*i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer1_list[2*i+1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

            hidden_conv1 = conv_layer2_list[0].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer2_list[1].get_output(input=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0,0], [0,0], [0,0], [32,32]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for i in range(1, n_layers):
                hidden_conv1 = conv_layer2_list[2*i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer2_list[2*i+1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

            hidden_conv1 = conv_layer3_list[0].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer3_list[1].get_output(input=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0,0], [0,0], [0,0], [64,64]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for i in range(1, n_layers):
                hidden_conv1 = conv_layer3_list[2*i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer3_list[2*i+1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)



            return hidden_conv

    def get_all_acc_list(self,test_images,test_labels,n_test,batch_size,is_train =True):
        batchs_number = 0
        hv_range = range(2, 10)

        # 子网准确率列表
        accuracy_1_list = []
        accuracy_1_1_list =[]
        accuracy_1_2_list = []
        accuracy_1_3_list = []

        accuracy_2_list = []
        accuracy_2_1_list = []
        accuracy_2_2_list = []
        accuracy_2_3_list = []


        accuracy_3_list = []

        # 不同决策的准确率列表
        acc_decision_batch_hv_dict={}
        for hv_number in hv_range:
            acc_decision_batch_hv_name = 'acc_decision_batch_hv%d' % hv_number # 合成键值名
            aborted_num_hv_name = 'aborted_num_hv%d' % hv_number  # 合成键值名
            acc_decision_batch_hv_dict[acc_decision_batch_hv_name] = []
            acc_decision_batch_hv_dict[aborted_num_hv_name] =0 #清0

        n_test =n_test - 1
        start_pos = 0
        if self.debug_mode:
            if self.is_training:
                start_pos = int(n_test *0.95)
            else:
                start_pos = int(n_test *0.98)
        for i in range(start_pos, n_test, batch_size):

            print('batchs_number: %d  , i: %d' % (batchs_number,i))
            batch_images = test_images[i: i + batch_size]
            batch_labels = test_labels[i: i + batch_size]

            [labels_array,
             avg_accuracy_1, avg_accuracy_1_1,avg_accuracy_1_2,avg_accuracy_1_3,
             avg_accuracy_2, avg_accuracy_2_1,avg_accuracy_2_2,avg_accuracy_2_3,
             avg_accuracy_3,
             logits_1, logits_1_1,logits_1_2,logits_1_3,
             logits_2, logits_2_1,logits_2_2,logits_2_3,
             logits_3
             ] = self.sess.run(
                fetches=[self.labels,
                         self.accuracy_1, self.accuracy_1_1,self.accuracy_1_2,self.accuracy_1_3,
                         self.accuracy_2, self.accuracy_2_1,self.accuracy_2_2,self.accuracy_2_3,
                         self.accuracy_3,
                         self.logits_1,self.logits_1_1,self.logits_1_2,self.logits_1_3,
                         self.logits_2,self.logits_2_1,self.logits_2_2,self.logits_2_3,
                         self.logits_3],
                feed_dict={self.images: batch_images,
                           self.labels: batch_labels,
                           self.keep_prob: 1.0})

            accuracy_1_list.append(avg_accuracy_1)
            accuracy_1_1_list.append(avg_accuracy_1_1)
            accuracy_1_2_list.append(avg_accuracy_1_2)
            accuracy_1_3_list.append(avg_accuracy_1_3)

            accuracy_2_list.append(avg_accuracy_2)
            accuracy_2_1_list.append(avg_accuracy_2_1)
            accuracy_2_2_list.append(avg_accuracy_2_2)
            accuracy_2_3_list.append(avg_accuracy_2_3)

            accuracy_3_list.append(avg_accuracy_3)

            predict_1 = self.sess.run(tf.argmax(logits_1, axis=1))
            predict_1_1 = self.sess.run(tf.argmax(logits_1_1, axis=1))
            predict_1_2 = self.sess.run(tf.argmax(logits_1_2, axis=1))
            predict_1_3 = self.sess.run(tf.argmax(logits_1_3, axis=1))

            predict_2 = self.sess.run(tf.argmax(logits_2, axis=1))
            predict_2_1 = self.sess.run(tf.argmax(logits_2_1, axis=1))
            predict_2_2 = self.sess.run(tf.argmax(logits_2_2, axis=1))
            predict_2_3 = self.sess.run(tf.argmax(logits_2_3, axis=1))

            predict_3 = self.sess.run(tf.argmax(logits_3, axis=1))

            # 几列预测值拼接成矩阵
            merrge_array = np.concatenate([[predict_1],[predict_1_1],[predict_1_2],[predict_1_3],
                                           [predict_2],[predict_2_1],[predict_2_2],[predict_2_3],
                                           [predict_3]
                                           ], axis=0)

            # 转置后，按一行一行比较
            merrge_array= np.transpose(merrge_array)

            (rows, cols) = merrge_array.shape

            final_batch_predict_dict={}
            for hv_number in hv_range:
                final_batch_predict_dict_name = 'final_batch_predict_hv%d' % hv_number
                # final_batch_predict_list_name = 'final_batch_predict_list_%d' % hv_number
                final_batch_predict_dict[final_batch_predict_dict_name] = []    #初始化清空


            delete_off_hv_dict ={}
            for hv_number in hv_range:
                delete_off_name = 'delete_off_v%d' % hv_number
                delete_off_hv_dict[delete_off_name] = 0  #初始化为0

            labels_array_dict ={}
            for hv_number in hv_range:
                labels_array_name = 'labels_array_hv%d' % hv_number
                labels_array_dict[labels_array_name] = labels_array    #加载label

            aborted_num_batch_hv_dict ={}
            for hv_number in hv_range:
                aborted_num_hv_name = 'aborted_num_hv%d' % hv_number
                aborted_num_batch_hv_dict[aborted_num_hv_name] = 0  #一个batch_size拒绝的次数    #初始化清空

            
            for row in range(0, rows):    #计算一批的
                result = all_np(merrge_array[row])  # 统计行个数  key:预测值  value：个数
                max_key = find_dict_max_key(result)  # 找到每行出现次数最多那个键值就是预测值
                for hv_number in hv_range:
                    if result[max_key] < hv_number:     # 如果达不到票数
                        labels_array_name = 'labels_array_hv%d' % hv_number #合成键值名
                        labels_array= labels_array_dict[labels_array_name]   #取出标签
                        delete_off_name = 'delete_off_v%d' % hv_number   #合成键值名
                        delete_off = delete_off_hv_dict[delete_off_name]   #取出删除偏移量
                        labels_array = np.delete(labels_array, row - delete_off, axis=0) #拒绝就要删对应标签，后面要做准确度对比
                        labels_array_dict[labels_array_name]=labels_array   #保存修改后的标签
                        delete_off_hv_dict[delete_off_name] = delete_off + 1   #保存删除偏移
                        aborted_num_hv_name = 'aborted_num_hv%d' % hv_number     #合成键值名
                        aborted_num_batch_hv_dict[aborted_num_hv_name] += 1        #记下本批拒绝次数

                    else:    # 如果达到票数
                        final_batch_predict_name = 'final_batch_predict_hv%d' % hv_number   #合成键值名

                        final_batch_list =final_batch_predict_dict[final_batch_predict_name]  #返回列表
                        final_batch_list.append(max_key)  #列表保存预测值
                        final_batch_predict_dict[final_batch_predict_name] =final_batch_list   #保存列表


            batchs_number = batchs_number + 1

            def get_acc_decision_batch(batch_labels_array,final_batch_predict_list):
                array_final_batch_predict_list = np.array(final_batch_predict_list)

                # 每一批的正确率都放到列表里面
                totol_batch_prediction = tf.equal(batch_labels_array, array_final_batch_predict_list)

                self.decision_batch_prediction = tf.reduce_mean(tf.cast(totol_batch_prediction, 'float'))
                acc_decision_batch = self.sess.run(self.decision_batch_prediction, feed_dict={self.labels: batch_labels})
                return acc_decision_batch

            def get_acc_decision_all_batch(hv_number):
                labels_array_name = 'labels_array_hv%d' % hv_number  # 合成键值名
                labels_array = labels_array_dict[labels_array_name]  # 取出标签
                final_batch_predict_name = 'final_batch_predict_hv%d' % hv_number  # 合成键值名
                final_batch_predict = final_batch_predict_dict[final_batch_predict_name]  # 取出批预测
                result_acc = get_acc_decision_batch(labels_array, final_batch_predict)  # 求出acc
                aborted_num_hv_name = 'aborted_num_hv%d' % hv_number  # 合成键值名
                reject_num = aborted_num_batch_hv_dict[aborted_num_hv_name]
                print('hv%d : %f, current aborted_num:%d ' % (hv_number, result_acc, reject_num))
                if np.isnan(result_acc) == False:
                    acc_decision_batch_hv_name = 'acc_decision_batch_hv%d' % hv_number
                    acc_list = acc_decision_batch_hv_dict[acc_decision_batch_hv_name]
                    acc_list.append(result_acc)
                    acc_decision_batch_hv_dict[acc_decision_batch_hv_name] = acc_list
                    acc_decision_batch_hv_dict[aborted_num_hv_name] += reject_num

            #统计所有批次
            threads_lists=[]
            for hv_number in hv_range:
                t = threading.Thread(target=get_acc_decision_all_batch,args=(hv_number,))
                threads_lists.append(t)

            for thread in threads_lists:
                thread.start()

            for thread in threads_lists:
                thread.join()

            print()

        return accuracy_1_list,accuracy_1_1_list,accuracy_1_2_list,accuracy_1_3_list, \
               accuracy_2_list,accuracy_2_1_list,accuracy_2_2_list,accuracy_2_3_list,\
               accuracy_3_list, \
               acc_decision_batch_hv_dict

        
    def train(self, dataloader, backup_path, n_epoch=5,
              batch_size=128,is_training =True):
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        # 构建会话
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=5)

        # 模型初始化
        # self.sess.run(tf.global_variables_initializer())
        try:
            print("\nTrying to restore last checkpoint ...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=backup_path)
            self.saver.restore(self.sess, save_path=last_chk_path)
            print("Restored checkpoint from:", last_chk_path)
        except ValueError:
            print("\nFailed to restore checkpoint. Initializing variables instead.")
            self.sess.run(tf.global_variables_initializer())
        
        # 验证集数据增强
        valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
        valid_labels = dataloader.valid_labels
        # 模型训练
        since = time.time()

        start_n_epoch= 0
        for epoch in range(start_n_epoch, n_epoch+1):

            # 训练集数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            
            # 开始本轮的训练，并计算目标函数值
            train_loss = 0.0
            get_global_step = 0
            start_pos=0
            if self.debug_mode:
                start_pos = int(dataloader.n_train * 0.98)
            for i in range(start_pos, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_loss, get_global_step] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss, self.global_step], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 0.5})
                if get_global_step % 20 == 0:
                    print('global_step: {} ,data_batch idx: {} , batch_loss: {}'.format(get_global_step, i, avg_loss))
                train_loss += avg_loss * batch_images.shape[0]
            # train_loss = 1.0 * train_loss / dataloader.n_train


            # 获取验证准确率列表
            if epoch % 5 == 0:
                accuracy_1_list,accuracy_1_1_list,accuracy_1_2_list,accuracy_1_3_list, \
                accuracy_2_list,accuracy_2_1_list,accuracy_2_2_list,accuracy_2_3_list, \
                accuracy_3_list, \
                acc_decision_batch_dict = \
                    self.get_all_acc_list(valid_images,valid_labels,dataloader.n_valid,batch_size,True)

                message_epoch = 'epoch: {} , global_step: {} \n'.format(epoch,get_global_step)
                message_1 = 'net1: %.4f' % (np.mean(accuracy_1_list))
                message_1_1 = ' net1_1: %.4f' % (np.mean(accuracy_1_1_list))
                message_1_2 = ' net1_2: %.4f' % (np.mean(accuracy_1_2_list))
                message_1_3 = ' net1_3: %.4f\n' % (np.mean(accuracy_1_3_list))

                message_2 = ' net2: %.4f' % (np.mean(accuracy_2_list))
                message_2_1 = ' net2_1: %.4f\n' % (np.mean(accuracy_2_1_list))
                message_2_2 = ' net2_2: %.4f\n' % (np.mean(accuracy_2_2_list))
                message_2_3 = ' net2_3: %.4f\n' % (np.mean(accuracy_2_3_list))
                

                message_3 = ' net3: %.4f\n' % (np.mean(accuracy_3_list))


                message_hv2 = ' acc_decision_batch_hv2: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv2']))
                message_hv3 = ' acc_decision_batch_hv3: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv3']))
                message_hv4 = ' acc_decision_batch_hv4: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv4']))
                message_hv5 = ' acc_decision_batch_hv5: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv5']))
                message_hv6 = ' acc_decision_batch_hv6: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv6']))
                message_hv7 = ' acc_decision_batch_hv7: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv7']))
                message_hv8 = ' acc_decision_batch_hv8: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv8']))
                message_hv9 = ' acc_decision_batch_hv9: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv9']))

                print_and_save_txt(str=message_epoch
                                       +message_1 + message_1_1 + message_1_2+ message_1_3
                                       +message_2+ message_2_1+message_2_2+message_2_3
                                       +message_3+
                                       message_hv2+message_hv3+message_hv4+message_hv5
                                        +message_hv6+message_hv7+message_hv8+message_hv9,
                                   filename=os.path.join(backup_path, 'train_log.txt'))
            # 保存模型
            if epoch % 10 == 0 :
                print('saving model.....')
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))

        #计算耗时
        time_elapsed = time.time()- since
        seconds = time_elapsed % 60
        hours = time_elapsed // 3600
        mins = (time_elapsed - hours * 3600) /3600 * 60
        time_message = 'The code run {:.0f}h {:.0f}m {:.0f}s\n'.format(
            hours,mins,seconds)
        print_and_save_txt(str=time_message,filename=os.path.join(backup_path, 'train_log.txt'))

        self.sess.close()
                
    def test(self, dataloader, backup_path, epoch, batch_size=128,is_training= False):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))


        test_images = dataloader.data_augmentation(dataloader.test_images,
                                                   flip=False, crop=True, crop_shape=(24, 24, 3), whiten=True,
                                                   noise=False)
        test_labels = dataloader.test_labels

        #获取全部准确率列表
        accuracy_1_list,accuracy_1_1_list,accuracy_1_2_list,accuracy_1_3_list,\
        accuracy_2_list,accuracy_2_1_list,accuracy_2_2_list,accuracy_2_3_list, \
        accuracy_3_list, \
        acc_decision_batch_dict =\
            self.get_all_acc_list(test_images, test_labels, dataloader.n_test, batch_size,is_train =False)


        message_1 = 'net1: %.4f' % (np.mean(accuracy_1_list))
        message_1_1 = ' net1_1: %.4f' % (np.mean(accuracy_1_1_list))
        message_1_2 = ' net1_2: %.4f' % (np.mean(accuracy_1_2_list))
        message_1_3 = ' net1_3: %.4f\n' % (np.mean(accuracy_1_3_list))

        message_2 = ' net2: %.4f' % (np.mean(accuracy_2_list))
        message_2_1 = ' net2_1: %.4f' % (np.mean(accuracy_2_1_list))
        message_2_2 = ' net2_2: %.4f' % (np.mean(accuracy_2_2_list))
        message_2_3 = ' net2_3: %.4f\n' % (np.mean(accuracy_2_3_list))

        message_3 = ' net3: %.4f\n' % (np.mean(accuracy_3_list))

        message_hv2 = ' acc_decision_batch_hv2: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv2']))
        message_hv3 = ' acc_decision_batch_hv3: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv3']))
        message_hv4 = ' acc_decision_batch_hv4: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv4']))
        message_hv5 = ' acc_decision_batch_hv5: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv5']))
        message_hv6 = ' acc_decision_batch_hv6: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv6']))
        message_hv7 = ' acc_decision_batch_hv7: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv7']))
        message_hv8 = ' acc_decision_batch_hv8: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv8']))
        message_hv9 = ' acc_decision_batch_hv9: %.4f\n' % (
            np.mean(acc_decision_batch_dict['acc_decision_batch_hv9']))
        print_and_save_txt(str=  message_1 + message_1_1 + message_1_2+message_1_3
                               + message_2 +message_2_1 +message_2_2  +message_2_3
                               +message_3
                               +message_hv2 + message_hv3 + message_hv4 + message_hv5
                               +message_hv6 + message_hv7 + message_hv8 + message_hv9,
                           filename=os.path.join(backup_path, 'test_log.txt'))


        print_and_save_txt('aborted_num_hv2 {} aborted_num_hv3 {} aborted_num_hv4 {}\n'
                           'aborted_num_hv5 {} aborted_num_hv6 {} aborted_num_hv7 {}\n'
                           'aborted_num_hv8 {} aborted_num_hv9 {}\n'
                           .format(acc_decision_batch_dict['aborted_num_hv2'],
                                acc_decision_batch_dict['aborted_num_hv3'],
                                acc_decision_batch_dict['aborted_num_hv4'],
                                acc_decision_batch_dict['aborted_num_hv5'],
                                acc_decision_batch_dict['aborted_num_hv6'],
                                acc_decision_batch_dict['aborted_num_hv7'],
                                acc_decision_batch_dict['aborted_num_hv8'],
                                acc_decision_batch_dict['aborted_num_hv9'])
                           ,filename=os.path.join(backup_path, 'test_log.txt'))

        #########  parameters numbers###########
        from functools import reduce
        from operator import mul
        def get_num_params():
            num_params = 0
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                num_params += reduce(mul, [dim.value for dim in shape], 1)
            return num_params

        print_and_save_txt(str='xxxxxxxxxxxxxx parament numbers is : %d xxxxxxxxxxxxxxx' % get_num_params(),
                           filename=os.path.join(backup_path, 'test_log.txt'))
        #######################################

        self.sess.close()
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.logits],
            feed_dict={self.images: numpy.random.random(size=[128, 24, 24, 3]),
                       self.labels: numpy.random.randint(low=0, high=9, size=[128,]),
                       self.keep_prob: 1.0})
        print(temp.shape)