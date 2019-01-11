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



class ConvNet():
    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=44,is_training =True):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers
        self.is_training  = is_training



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
        hidden_conv_1 = self.residual_inference(images=basic_conv, scope_name='net_1_1')
        hidden_conv_1 = self.son_google_v2_part(hidden_conv_1,'net_1_2')
        # global average pooling
        input_dense_1 = tf.reduce_mean(hidden_conv_1, reduction_indices=[1, 2])
        self.logits_1 = dense_layer_1.get_output(input=input_dense_1)

        #子层1------孙一 densenet
        self.logits_1_1 = self.grand_son_1(hidden_conv_1,scope_name='grand_son_1')
        # global average pooling
        #         input_dense_1_1 = tf.reduce_mean(hidden_conv_1_1, reduction_indices=[1, 2])
        # self.logits_1_1 = dense_layer_1.get_output(input=input_dense_1_1)

        #子层2
        hidden_conv_2 = self.residual_inference(images=basic_conv, scope_name='net_2_1')
        hidden_conv_2 = self.son_google_v3_part(hidden_conv_2,'net_2_2')
        # global average pooling
        input_dense_2 = tf.reduce_mean(hidden_conv_2, reduction_indices=[1, 2])
        self.logits_2 = dense_layer_2.get_output(input=input_dense_2)

        # 子层3
        hidden_conv_3 = self.residual_inference(images=basic_conv, scope_name='net_3_1')
        hidden_conv_3 =self.son_cnn_part(hidden_conv_3,scope_name='net_3_2')
        # global average pooling
        input_dense_3= tf.reduce_mean(hidden_conv_3, reduction_indices=[1, 2])
        self.logits_3 = dense_layer_3.get_output(input=input_dense_3)

        # 目标函数
        self.objective_1 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_1, labels=self.labels))

        self.objective_1_1 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_1_1, labels=self.labels))


        self.objective_2 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_2, labels=self.labels))

        self.objective_3 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_3, labels=self.labels))

        # self.objective_4 = tf.reduce_sum(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=self.logits_4, labels=self.labels))
        #
        # self.objective_5 = tf.reduce_sum(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(
        #         logits=self.logits_5, labels=self.labels))

        self.objective = self.objective_1 + self.objective_2 + \
                         self.objective_3 +self.objective_1_1

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

        # 观察值 grand son1
        correct_prediction_1_1 = tf.equal(self.labels, tf.argmax(self.logits_1_1, 1))
        self.correct_prediction_1_1 = tf.reduce_mean(tf.cast(correct_prediction_1_1, 'float'))

        # son2
        correct_prediction_2 = tf.equal(self.labels, tf.argmax(self.logits_2, 1))
        self.accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, 'float'))

        # son3
        correct_prediction_3 = tf.equal(self.labels, tf.argmax(self.logits_3, 1))
        self.accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, 'float'))

        # correct_prediction_4 = tf.equal(self.labels, tf.argmax(self.logits_4, 1))
        # self.accuracy_4 = tf.reduce_mean(tf.cast(correct_prediction_4, 'float'))
        #
        # correct_prediction_5 = tf.equal(self.labels, tf.argmax(self.logits_5, 1))
        # self.accuracy_5 = tf.reduce_mean(tf.cast(correct_prediction_5, 'float'))


    def grand_son_1(self,hidden_conv_1,scope_name):
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

                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_1, self.n_classes, self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def son_cnn_part(self, hidden_conv, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                hidden_conv = slim.conv2d(hidden_conv, depth(200), [3, 3], scope='Conv2d_1')
        return hidden_conv

    def son_google_v3_part(self, hidden_conv, scope_name):
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

    def trainsition_layer_to_classes(self,_input, n_classes=10, training=True):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        _output = output = slim.batch_norm(_input, is_training=training)
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

    def get_5_acc_list(self,test_images,test_labels,n_test,batch_size,is_train =True):
        # 子网准确率列表
        accuracy_1_list = []
        accuracy_1_1_list =[]

        accuracy_2_list = []
        accuracy_3_list = []
        # accuracy_4_list = []
        accuracy_5_list = []

        # 不同决策的准确率列表
        acc_decision_batch_hv2 = []
        acc_decision_batch_hv3 = []
        acc_decision_batch_hv4 = []
        acc_decision_batch_hv5 = []

        #不预测的总数量

        aborted_num_total_hv2 = 0
        aborted_num_total_hv3 = 0
        aborted_num_total_hv4 = 0
        aborted_num_total_hv5 = 0

        # acc_decision_batchs = numpy.float32(.0)

        batchs_number = 0
        for i in range(0, n_test, batch_size):
            print('batch_size %d '%i)
            batch_images = test_images[i: i + batch_size]
            batch_labels = test_labels[i: i + batch_size]

            [labels_array,
             avg_accuracy_1, avg_accuracy_1_1,
             avg_accuracy_2, avg_accuracy_3,
             logits_1, logits_1_1,
             logits_2, logits_3] = self.sess.run(
                fetches=[self.labels,
                         self.accuracy_1, self.correct_prediction_1_1,
                         self.accuracy_2, self.accuracy_3,
                         self.logits_1,self.logits_1_1,
                         self.logits_2, self.logits_3],
                feed_dict={self.images: batch_images,
                           self.labels: batch_labels,
                           self.keep_prob: 1.0})

            accuracy_1_list.append(avg_accuracy_1)
            accuracy_1_1_list.append(avg_accuracy_1_1)

            accuracy_2_list.append(avg_accuracy_2)
            accuracy_3_list.append(avg_accuracy_3)
            # accuracy_4_list.append(avg_accuracy_4)
            # accuracy_5_list.append(avg_accuracy_5)

            predict_1 = self.sess.run(tf.argmax(logits_1, axis=1))
            predict_1_1 = self.sess.run(tf.argmax(logits_1_1, axis=1))

            predict_2 = self.sess.run(tf.argmax(logits_2, axis=1))
            predict_3 = self.sess.run(tf.argmax(logits_3, axis=1))
            # predict_4 = self.sess.run(tf.argmax(logits_4, axis=1))
            # predict_5 = self.sess.run(tf.argmax(logits_5, axis=1))

            # 几列预测值拼接成矩阵
            merrge_array = np.concatenate([[predict_1],[predict_1_1],
                                           [predict_2], [predict_3]
                                           ], axis=0)

            # 转置后，按一行一行比较
            merrge_array= np.transpose(merrge_array)

            (rows, cols) = merrge_array.shape
            final_batch_predict_hv2 = []
            final_batch_predict_hv3 = []
            final_batch_predict_hv4 = []
            # final_batch_predict_hv5 = []


            delete_off_v2 = 0
            delete_off_v3 = 0
            delete_off_v4 = 0
            # delete_off_v5 = 0


            labels_array_hv2 = labels_array
            labels_array_hv3 = labels_array
            labels_array_hv4 = labels_array
            # labels_array_hv5 = labels_array

            aborted_num_hv2 = 0
            aborted_num_hv3 = 0
            aborted_num_hv4 = 0
            # aborted_num_hv5 = 0
            for row in range(0, rows):
                result = all_np(merrge_array[row])  # 统计行个数
                max_key = find_dict_max_key(result)  # 找到每行出现次数最多那个键值就是预测值
                if result[max_key] <2 :
                    labels_array_hv2 = np.delete(labels_array_hv2,row-delete_off_v2,axis=0)
                    delete_off_v2 +=1
                    aborted_num_hv2 +=1
                else:
                    final_batch_predict_hv2.append(max_key)  # 预测值存到列表里面

                if result[max_key] <3:
                    labels_array_hv3 = np.delete(labels_array_hv3, row - delete_off_v3, axis=0)
                    delete_off_v3 += 1
                    aborted_num_hv3 += 1
                else:
                    final_batch_predict_hv3.append(max_key)  # 预测值存到列表里面

                if result[max_key] < 4:
                    labels_array_hv4 = np.delete(labels_array_hv4, row - delete_off_v4, axis=0)
                    delete_off_v4 += 1
                    aborted_num_hv4 += 1
                else:
                    final_batch_predict_hv4.append(max_key)  # 预测值存到列表里面

                # if result[max_key] < 5:
                #     labels_array_hv5 = np.delete(labels_array_hv5, row - delete_off_v5, axis=0)
                #     delete_off_v5 += 1
                #     aborted_num_hv5 += 1
                # else:
                #     final_batch_predict_hv5.append(max_key)  # 预测值存到列表里面

            batchs_number = batchs_number + 1
            # 列表转数组
            def get_acc_decision_batch(batch_labels_array,final_batch_predict_list):
                array_final_batch_predict_list = np.array(final_batch_predict_list)

                # 每一批的正确率都放到列表里面
                totol_batch_prediction = tf.equal(batch_labels_array, array_final_batch_predict_list)

                self.decision_batch_prediction = tf.reduce_mean(tf.cast(totol_batch_prediction, 'float'))
                acc_decision_batch = self.sess.run(self.decision_batch_prediction, feed_dict={self.labels: batch_labels})
                return acc_decision_batch

            result2 = get_acc_decision_batch(
                labels_array_hv2, final_batch_predict_hv2)
            print('result2 %f aborted_num:%d '% (result2,aborted_num_hv2))
            aborted_num_total_hv2 += aborted_num_hv2  # 计算抛弃预测数量
            acc_decision_batch_hv2.append(result2)

            result3 = get_acc_decision_batch(
                labels_array_hv3, final_batch_predict_hv3)
            print('result3 %f aborted_num:%d ' % (result3,aborted_num_hv3))
            aborted_num_total_hv3 += aborted_num_hv3  # 计算抛弃预测数量
            acc_decision_batch_hv3.append(result3)

            result4 = get_acc_decision_batch(
                labels_array_hv4, final_batch_predict_hv4)
            print('result4 %f aborted_num:%d ' % (result4,aborted_num_hv4))
            aborted_num_total_hv4 += aborted_num_hv4 #计算抛弃预测数量
            acc_decision_batch_hv4.append(result4)

            # result5 = get_acc_decision_batch(
            #     labels_array_hv5, final_batch_predict_hv5)
            # print('result5 %f aborted_num:%d ' % (result5, aborted_num_hv5))
            # aborted_num_total_hv5 += aborted_num_hv5 #计算抛弃预测数量
            # acc_decision_batch_hv5.append(result5)

        acc_decision_batch_dict = {}
        acc_decision_batch_dict['acc_decision_batch_hv2']= acc_decision_batch_hv2
        acc_decision_batch_dict['acc_decision_batch_hv3']= acc_decision_batch_hv3
        acc_decision_batch_dict['acc_decision_batch_hv4']= acc_decision_batch_hv4
        # acc_decision_batch_dict['acc_decision_batch_hv5']= acc_decision_batch_hv5

        acc_decision_batch_dict['aborted_num_hv2'] =aborted_num_total_hv2
        acc_decision_batch_dict['aborted_num_hv3'] =aborted_num_total_hv3
        acc_decision_batch_dict['aborted_num_hv4'] =aborted_num_total_hv4
        # acc_decision_batch_dict['aborted_num_hv5'] =aborted_num_total_hv5

        # if not is_train:
        # print('batches: {} , acc_decision_batch: hv2{}   hv3{}   hv4{}   hv5{}'.format(batchs_number,
        # acc_decision_batch_hv2,acc_decision_batch_hv3,acc_decision_batch_hv4,acc_decision_batch_hv5))


        # return accuracy_1_list,accuracy_2_list,accuracy_3_list, \
        #        accuracy_4_list,accuracy_5_list,acc_decision_batch_dict

        return accuracy_1_list,accuracy_1_1_list, \
               accuracy_2_list, accuracy_3_list, \
         acc_decision_batch_dict

        
    def train(self, dataloader, backup_path, n_epoch=5,
              batch_size=128,is_training =True):
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
            for i in range(0, dataloader.n_train, batch_size):
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
            train_loss = 1.0 * train_loss / dataloader.n_train


            # 获取验证准确率列表
            if epoch % 5 == 0:
                accuracy_1_list,accuracy_1_1_list, accuracy_2_list, accuracy_3_list, \
                acc_decision_batch_dict = \
                    self.get_5_acc_list(valid_images,valid_labels,dataloader.n_valid,batch_size,True)

                message_1 = 'epoch: {} , global_step: {} \n'.format(epoch,get_global_step)
                message_2 = 'net1: %.4f' % (np.mean(accuracy_1_list))
                message_5 = ' net1_1: %.4f' % (np.mean(accuracy_1_1_list))

                message_3 = ' net2: %.4f' % (np.mean(accuracy_2_list))
                message_4 = ' net3: %.4f' % (np.mean(accuracy_3_list))

                # message_5 = ' net4: %.4f' % (np.mean(accuracy_4_list))
                # message_6 = ' net5: %.4f' % (np.mean(accuracy_5_list))
                message_7 = ' acc_decision_batch_hv2: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv2']))
                message_8 = ' acc_decision_batch_hv3: %.4f\n' % (
                    np.mean(acc_decision_batch_dict['acc_decision_batch_hv3']))
                # message_9 = ' acc_decision_batch_hv4: %.4f\n' % (
                #     np.mean(acc_decision_batch_dict['acc_decision_batch_hv4']))
                # message_10 = ' acc_decision_batch_hv5: %.4f\n' % (
                #     np.mean(acc_decision_batch_dict['acc_decision_batch_hv5']))
                # print_and_save_txt(str=message_1 + message_2 + message_3 + message_4 + message_5
                #                        + message_6 + message_7 + message_8 + message_9 + message_10,
                #                    filename=os.path.join(backup_path, 'train_log.txt'))

                print_and_save_txt(str=message_1 + message_2 + message_3 + message_4 +
                                       message_5+message_7,
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
        accuracy_1_list,accuracy_1_1_list,\
        accuracy_2_list, accuracy_3_list, \
        acc_decision_batch_dict =\
            self.get_5_acc_list(test_images, test_labels, dataloader.n_test, batch_size,is_train =False)

        message_1 = 'test result: \n'
        message_2 = 'net1: %.4f' % (np.mean(accuracy_1_list))
        message_3 = ' net2: %.4f' % (np.mean(accuracy_2_list))
        message_4 = ' net3: %.4f' % (np.mean(accuracy_3_list))
        message_5 = ' net1_1: %.4f' % (np.mean(accuracy_1_1_list))
        # message_6 = ' net4: %.4f' % (np.mean(accuracy_5_list))
        message_7 = ' acc_decision_batch_hv2: %.4f\n' % (np.mean(acc_decision_batch_dict['acc_decision_batch_hv2']))
        message_8 = ' acc_decision_batch_hv3: %.4f\n' % (np.mean(acc_decision_batch_dict['acc_decision_batch_hv3']))
        # message_9 = ' acc_decision_batch_hv4: %.4f\n' % (np.mean(acc_decision_batch_dict['acc_decision_batch_hv4']))
        # message_10 = ' acc_decision_batch_hv5: %.4f\n' % (np.mean(acc_decision_batch_dict['acc_decision_batch_hv5']))
        # print_and_save_txt(str=message_1 + message_2 + message_3 + message_4 + message_5
        #                    +message_6+message_7+message_8+message_9+message_10,
        #                    filename=os.path.join(backup_path, 'test_log.txt'))

        print_and_save_txt(str=message_1 + message_2 + message_3 + message_4+message_5
                           +message_7+message_8,
                           filename=os.path.join(backup_path, 'test_log.txt'))


        # print_and_save_txt('aborted_num_hv2 {} aborted_num_hv3 {} aborted_num_hv4 {} aborted_num_hv5 {}\n'
        #                    .format(acc_decision_batch_dict['aborted_num_hv2'],
        #                            acc_decision_batch_dict['aborted_num_hv3'],
        #                            acc_decision_batch_dict['aborted_num_hv4'],
        #                            acc_decision_batch_dict['aborted_num_hv5']),
        #                    filename=os.path.join(backup_path, 'test_log.txt'))

        print_and_save_txt('aborted_num_hv2 {} aborted_num_hv3 ｛｝aborted_num_hv4 ｛｝\n'
                           .format(acc_decision_batch_dict['aborted_num_hv2'],
                                   acc_decision_batch_dict['aborted_num_hv3']),
                                    acc_decision_batch_dict['aborted_num_hv4'],
                           filename=os.path.join(backup_path, 'test_log.txt'))

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