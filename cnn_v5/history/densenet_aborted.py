import tensorflow as tf
import numpy as np
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


depth = 40
growthRate = 12

def conv(name, l, channel, stride):
    return Conv2D(name, l, channel, 3, stride=stride,
                  nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)))


def add_layer(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = BatchNorm(l)
        c = tf.nn.relu(c)
        c = conv('conv1', c, growthRate, 1)
        l = tf.concat([c, l], 3)
    return l


def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = BatchNorm( l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
    return l

def dense_net(input, input_shape,class_num):
    end_point = []
    x_image = tf.reshape(input, input_shape)  # 转换输入数据shape,以便于用于网络中
    N=  int((depth - 4)  / 3)


    l = conv('conv0', x_image, 16, 1)

    with tf.variable_scope('block1') as scope:
        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition1', l)

    with tf.variable_scope('block2') as scope:
        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
        l = add_transition('transition2', l)

    with tf.variable_scope('block3') as scope:
        for i in range(N):
            l = add_layer('dense_layer.{}'.format(i), l)
    l = BatchNorm(l)
    l = tf.nn.relu(l)
    l = GlobalAvgPooling('gap', l)
    y_predict = FullyConnected('linear', l, out_dim=class_num, nl=tf.identity)
    return y_predict,end_point