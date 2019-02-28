import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import cfg
import numpy as np


def cross_ent_loss(output, x, y):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=output)
    loss = tf.reduce_mean(loss)
    num_class = int(output.get_shape()[-1])
    data_size = int(x.get_shape()[1])
    data_channel = int(x.get_shape()[3])

    # reconstruction loss
    y = tf.one_hot(y, num_class, dtype=tf.float32)
    y = tf.expand_dims(y, axis=2)
    output = tf.expand_dims(output, axis=2)
    output = tf.reshape(tf.multiply(output, y), shape=[cfg.batch_size, -1])
    tf.logging.info("decoder input value dimension:{}".format(output.get_shape()))

    with tf.variable_scope('decoder'):
        output = slim.fully_connected(output, 512, trainable=True)
        output = slim.fully_connected(output, 1024, trainable=True)
        output = slim.fully_connected(output, data_size * data_size *data_channel,
                                      trainable=True, activation_fn=tf.sigmoid)

        x = tf.reshape(x, shape=[cfg.batch_size, -1])
        reconstruction_loss = tf.reduce_mean(tf.square(output - x))

    # regularization loss
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # loss+0.0005*reconstruction_loss+regularization#
    loss_all = tf.add_n([loss] + [0.0005 * reconstruction_loss] + regularization)

    return loss_all, reconstruction_loss, output


def build_arch_baseline(input, is_train: bool, num_classes: int):

    bias_initializer = tf.truncated_normal_initializer(
        mean=0.0, stddev=0.01)  # tf.constant_initializer(0.0)
    # The paper didnot mention any regularization, a common l2 regularizer to weights is added here
    weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)

    tf.logging.info('input shape: {}'.format(input.get_shape()))

    # weights_initializer=initializer,
    with slim.arg_scope([slim.conv2d, slim.fully_connected], trainable=is_train, biases_initializer=bias_initializer, weights_regularizer=weights_regularizer):
        with tf.variable_scope('relu_conv1') as scope:
            output = slim.conv2d(input, num_outputs=32, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer1')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        with tf.variable_scope('relu_conv2') as scope:
            output = slim.conv2d(output, num_outputs=64, kernel_size=[
                                 5, 5], stride=1, padding='SAME', scope=scope, activation_fn=tf.nn.relu)
            output = slim.max_pool2d(output, [2, 2], scope='max_2d_layer2')

            tf.logging.info('output shape: {}'.format(output.get_shape()))

        output = slim.flatten(output)
        output = slim.fully_connected(output, 1024, scope='relu_fc3', activation_fn=tf.nn.relu)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        output = slim.dropout(output, 0.5, scope='dp')
        output = slim.fully_connected(output, num_classes, scope='final_layer', activation_fn=None)
        tf.logging.info('output shape: {}'.format(output.get_shape()))
        return output


def test_accuracy(logits, labels):
    # logits = tf.argmax(logits, 1)
    logits_idx = tf.to_int32(tf.argmax(logits, axis=1))
    logits_idx = tf.reshape(logits_idx, shape=(cfg.batch_size,))
    correct_preds = tf.equal(tf.to_int32(labels), logits_idx)
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) / cfg.batch_size

    return accuracy