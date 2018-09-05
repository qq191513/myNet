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

import operator
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


def get_sess(backup_path, epoch):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # 读取模型
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
    assert (os.path.exists(model_path + '.index'))
    saver.restore(sess, model_path)
    print('read model from %s' % (model_path))
    return sess


def get_all_acc_list(sess_list,ConvNet_list,setting,test_images, test_labels, n_test, batch_size, is_train=True):
    def get_acc_decision_batch(ConvNet,batch_labels_array, final_batch_predict_list):
        array_final_batch_predict_list = np.array(final_batch_predict_list)

        # 每一批的正确率都放到列表里面
        totol_batch_prediction = tf.equal(batch_labels_array, array_final_batch_predict_list)

        decision_batch_prediction = tf.reduce_mean(tf.cast(totol_batch_prediction, 'float'))
        acc_decision_batch = sess.run(decision_batch_prediction, feed_dict={ConvNet.labels: batch_labels_array})
        return acc_decision_batch

    def soft_max_predict(ConvNet,batch_images, batch_labels):
        # 算出所有logits的概率矩阵

        probability_value = {}
        for name, op in ConvNet.probability_op.items():
            probability_value[name] = sess.run(
                op, feed_dict={ConvNet.images: batch_images,
                               ConvNet.keep_prob: 1.0})

        # 把每个网络输出的概率矩阵相加，每个概率矩阵为batch_size * num_class
        sum = 0
        for name, value in probability_value.items():
            sum = sum + value

        rows, cols = sum.shape
        delete_number = 0
        dele_row_list = []
        for row in range(0, rows):
            max_value = np.max(sum[row])
            if max_value < 8:
                dele_row_list.append(row)
                delete_number += 1
        sum = np.delete(sum, dele_row_list, axis=0)
        batch_labels = np.delete(batch_labels, dele_row_list, axis=0)

        predict_batch = sess.run(tf.argmax(sum, axis=1))
        soft_result_acc = get_acc_decision_batch(ConvNet,batch_labels, predict_batch)
        print('soft predict result_acc : %.4f  , delete_number %d ' % (soft_result_acc, delete_number))
        return soft_result_acc, delete_number


    batchs_number = 0
    hv_range = range(2, 13)


    soft_result_acc_list = []
    # 不同决策的准确率列表
    acc_decision_batch_hv_dict = {}
    for hv_number in hv_range:
        acc_decision_batch_hv_name = 'acc_decision_batch_hv%d' % hv_number  # 合成键值名
        aborted_num_hv_name = 'aborted_num_hv%d' % hv_number  # 合成键值名
        acc_decision_batch_hv_dict[acc_decision_batch_hv_name] = []
        acc_decision_batch_hv_dict[aborted_num_hv_name] = 0  # 清0

    n_test = n_test - n_test % batch_size
    start_pos = 0
    soft_delete_number_total = 0
    if setting.only_test_small_part_dataset:
        start_pos = int(n_test * setting.test_proprotion)
    if setting.debug_mode:
        start_pos = int(n_test * setting.test_proprotion)

    for i in range(start_pos, n_test, batch_size):

        print('batchs_number: %d  , i: %d' % (batchs_number, i))
        batch_images = test_images[i: i + batch_size]
        batch_labels = test_labels[i: i + batch_size]

        order = 1
        sess_dict={}
        for sess,ConvNet in zip(sess_list,ConvNet_list):
            sess_name = 'sess_%d' % (order)
            sess_name_acc_list = sess_name + 'acc_list'
            order = order + 1
            [labels_array,
             avg_accuracy_1, avg_accuracy_1_1, avg_accuracy_1_2, avg_accuracy_1_3,
             avg_accuracy_2, avg_accuracy_2_1, avg_accuracy_2_2, avg_accuracy_2_3,
             avg_accuracy_3, avg_accuracy_3_1, avg_accuracy_3_2, avg_accuracy_3_3,
             logits_1, logits_1_1, logits_1_2, logits_1_3,
             logits_2, logits_2_1, logits_2_2, logits_2_3,
             logits_3, logits_3_1, logits_3_2, logits_3_3,
             ] = sess.run(
                fetches=[ConvNet.labels,
                         ConvNet.accuracy_1, ConvNet.accuracy_1_1, ConvNet.accuracy_1_2, ConvNet.accuracy_1_3,
                         ConvNet.accuracy_2, ConvNet.accuracy_2_1, ConvNet.accuracy_2_2, ConvNet.accuracy_2_3,
                         ConvNet.accuracy_3, ConvNet.accuracy_3_1, ConvNet.accuracy_3_2, ConvNet.accuracy_3_3,
                         ConvNet.logits_1, ConvNet.logits_1_1, ConvNet.logits_1_2, ConvNet.logits_1_3,
                         ConvNet.logits_2, ConvNet.logits_2_1, ConvNet.logits_2_2, ConvNet.logits_2_3,
                         ConvNet.logits_3, ConvNet.logits_3_1, ConvNet.logits_3_2, ConvNet.logits_3_3,
                         ],
                feed_dict={ConvNet.images: batch_images,
                           ConvNet.labels: batch_labels,
                           ConvNet.keep_prob: 1.0})


            soft_result_acc, soft_delete_number = soft_max_predict(ConvNet,batch_images, labels_array)
            soft_result_acc_list.append(soft_result_acc)
            soft_delete_number_total += soft_delete_number


            ConvNet.accuracy_1_list.append(avg_accuracy_1)
            ConvNet.accuracy_1_1_list.append(avg_accuracy_1_1)
            ConvNet.accuracy_1_2_list.append(avg_accuracy_1_2)
            ConvNet.accuracy_1_3_list.append(avg_accuracy_1_3)

            ConvNet.accuracy_2_list.append(avg_accuracy_2)
            ConvNet.accuracy_2_1_list.append(avg_accuracy_2_1)
            ConvNet.accuracy_2_2_list.append(avg_accuracy_2_2)
            ConvNet.accuracy_2_3_list.append(avg_accuracy_2_3)

            ConvNet.accuracy_3_list.append(avg_accuracy_3)
            ConvNet.accuracy_3_1_list.append(avg_accuracy_3_1)
            ConvNet.accuracy_3_2_list.append(avg_accuracy_3_2)
            ConvNet.accuracy_3_3_list.append(avg_accuracy_3_3)

            sess_dict[sess_name_acc_list] = [ConvNet.accuracy_1_list,ConvNet.accuracy_1_1_list,ConvNet.accuracy_1_2_list,ConvNet.accuracy_1_3_list,
                                             ConvNet.accuracy_2_list,ConvNet.accuracy_2_1_list,ConvNet.accuracy_2_2_list,ConvNet.accuracy_2_3_list,
                                             ConvNet.accuracy_3_list,ConvNet.accuracy_3_1_list,ConvNet.accuracy_3_2_list,ConvNet.accuracy_3_3_list,
                                             ]

            predict_1 = sess.run(tf.argmax(logits_1, axis=1))
            predict_1_1 = sess.run(tf.argmax(logits_1_1, axis=1))
            predict_1_2 = sess.run(tf.argmax(logits_1_2, axis=1))
            predict_1_3 = sess.run(tf.argmax(logits_1_3, axis=1))

            predict_2 = sess.run(tf.argmax(logits_2, axis=1))
            predict_2_1 = sess.run(tf.argmax(logits_2_1, axis=1))
            predict_2_2 = sess.run(tf.argmax(logits_2_2, axis=1))
            predict_2_3 = sess.run(tf.argmax(logits_2_3, axis=1))

            predict_3 = sess.run(tf.argmax(logits_3, axis=1))
            predict_3_1 = sess.run(tf.argmax(logits_3_1, axis=1))
            predict_3_2 = sess.run(tf.argmax(logits_3_2, axis=1))
            predict_3_3 = sess.run(tf.argmax(logits_3_3, axis=1))

            sess_name_predict = sess_name + '_predict'
            sess_dict[sess_name_predict] =[[predict_1], [predict_1_1], [predict_1_2], [predict_1_3],
                                       [predict_2], [predict_2_1], [predict_2_2], [predict_2_3],
                                       [predict_3], [predict_3_1], [predict_3_2], [predict_3_3],
                                       ]
        # 几列预测值拼接成矩阵
        merrge_array = np.concatenate(sess_dict['sess_1_predict'],sess_dict['sess_2_predict'],axis=1)

        # merrge_array = np.concatenate([[predict_1], [predict_1_1], [predict_1_2], [predict_1_3],
        #                                [predict_2], [predict_2_1], [predict_2_2], [predict_2_3],
        #                                [predict_3], [predict_3_1], [predict_3_2], [predict_3_3],
        #                                ], axis=0)
        merrge_array = np.concatenate(merrge_array,axis=0)

        # 转置后，按一行一行比较
        merrge_array = np.transpose(merrge_array)

        (rows, cols) = merrge_array.shape

        final_batch_predict_dict = {}
        for hv_number in hv_range:
            final_batch_predict_dict_name = 'final_batch_predict_hv%d' % hv_number
            final_batch_predict_dict[final_batch_predict_dict_name] = []  # 初始化清空

        delete_off_hv_dict = {}
        for hv_number in hv_range:
            delete_off_name = 'delete_off_v%d' % hv_number
            delete_off_hv_dict[delete_off_name] = 0  # 初始化为0

        labels_array_dict = {}
        for hv_number in hv_range:
            labels_array_name = 'labels_array_hv%d' % hv_number
            labels_array_dict[labels_array_name] = labels_array  # 加载label

        aborted_num_batch_hv_dict = {}
        for hv_number in hv_range:
            aborted_num_hv_name = 'aborted_num_hv%d' % hv_number
            aborted_num_batch_hv_dict[aborted_num_hv_name] = 0  # 一个batch_size拒绝的次数    #初始化清空

        for row in range(0, rows):  # 计算一批的
            result = all_np(merrge_array[row])  # 统计行个数  key:预测值  value：个数
            max_key = find_dict_max_key(result)  # 找到每行出现次数最多那个键值就是预测值
            for hv_number in hv_range:
                if result[max_key] < hv_number:  # 如果达不到票数
                    labels_array_name = 'labels_array_hv%d' % hv_number  # 合成键值名
                    labels_array = labels_array_dict[labels_array_name]  # 取出标签
                    delete_off_name = 'delete_off_v%d' % hv_number  # 合成键值名
                    delete_off = delete_off_hv_dict[delete_off_name]  # 取出删除偏移量
                    labels_array = np.delete(labels_array, row - delete_off, axis=0)  # 拒绝就要删对应标签，后面要做准确度对比
                    labels_array_dict[labels_array_name] = labels_array  # 保存修改后的标签
                    delete_off_hv_dict[delete_off_name] = delete_off + 1  # 保存删除偏移
                    aborted_num_hv_name = 'aborted_num_hv%d' % hv_number  # 合成键值名
                    aborted_num_batch_hv_dict[aborted_num_hv_name] += 1  # 记下本批拒绝次数

                else:  # 如果达到票数
                    final_batch_predict_name = 'final_batch_predict_hv%d' % hv_number  # 合成键值名

                    final_batch_list = final_batch_predict_dict[final_batch_predict_name]  # 返回列表
                    final_batch_list.append(max_key)  # 列表保存预测值
                    final_batch_predict_dict[final_batch_predict_name] = final_batch_list  # 保存列表

        batchs_number = batchs_number + 1

        # 统计所有批次
        for hv_number in hv_range:
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

        # print('finsoft_result_acc: ',soft_result_acc_list)
    acc_decision_batch_hv_dict['soft_delete_number_total'] = soft_delete_number_total
    # return accuracy_1_list, accuracy_1_1_list, accuracy_1_2_list, accuracy_1_3_list, \
    #        accuracy_2_list, accuracy_2_1_list, accuracy_2_2_list, accuracy_2_3_list, \
    #        accuracy_3_list, accuracy_3_1_list, accuracy_3_2_list, accuracy_3_3_list, \
    #        acc_decision_batch_hv_dict, soft_result_acc_list
    return sess_dict,acc_decision_batch_hv_dict, soft_result_acc_list

def test(sess_list, ConvNet_list,setting,dataloader, backup_path, batch_size=128,is_training = False):
    test_images = dataloader.data_augmentation(dataloader.test_images,
                                               flip=False, crop=True, crop_shape=(24, 24, 3), whiten=True,
                                               noise=False)
    test_labels = dataloader.test_labels

    # 获取全部准确率列表
    # accuracy_1_list, accuracy_1_1_list, accuracy_1_2_list, accuracy_1_3_list, \
    # accuracy_2_list, accuracy_2_1_list, accuracy_2_2_list, accuracy_2_3_list, \
    # accuracy_3_list, accuracy_3_1_list, accuracy_3_2_list, accuracy_3_3_list, \
    sess_dict,acc_decision_batch_dict, soft_result_acc_list = \
        get_all_acc_list(sess_list,ConvNet_list,setting,test_images, test_labels, dataloader.n_test, batch_size, is_train=is_training)
    for key,sess_acc_list in sess_dict.item():

        message_1 = '%s_net1: %.4f' % (key,np.mean(sess_acc_list.accuracy_1_list))
        message_1_1 = ' %s_net1_1: %.4f' % (key,np.mean(sess_acc_list.accuracy_1_1_list))
        message_1_2 = ' %s_net1_2: %.4f' % (key,np.mean(sess_acc_list.accuracy_1_2_list))
        message_1_3 = ' %s_net1_3: %.4f\n' % (key,np.mean(sess_acc_list.accuracy_1_3_list))

        message_2 = ' %s_net2: %.4f' % (key,np.mean(sess_acc_list.accuracy_2_list))
        message_2_1 = ' %s_net2_1: %.4f' % (key,np.mean(sess_acc_list.accuracy_2_1_list))
        message_2_2 = ' %s_net2_2: %.4f' % (key,np.mean(sess_acc_list.accuracy_2_2_list))
        message_2_3 = ' %s_net2_3: %.4f\n' % (key,np.mean(sess_acc_list.accuracy_2_3_list))

        message_3 = ' %s_net3: %.4f' % (key,np.mean(sess_acc_list.accuracy_3_list))
        message_3_1 = ' %s_net3_1: %.4f' % (key,np.mean(sess_acc_list.accuracy_3_1_list))
        message_3_2 = ' %s_net3_2: %.4f' % (key,np.mean(sess_acc_list.accuracy_3_2_list))
        message_3_3 = ' %s_net3_3: %.4f\n' % (key,np.mean(sess_acc_list.accuracy_3_3_list))



        print_and_save_txt(str=message_1 + message_1_1 + message_1_2 + message_1_3
                               + message_2 + message_2_1 + message_2_2 + message_2_3
                               + message_3 + message_3_1 + message_3_2 + message_3_3,
                            filename=os.path.join(backup_path, 'test_log.txt'))

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
    message_hv10 = ' acc_decision_batch_hv10: %.4f\n' % (
        np.mean(acc_decision_batch_dict['acc_decision_batch_hv10']))
    message_hv11 = ' acc_decision_batch_hv11: %.4f\n' % (
        np.mean(acc_decision_batch_dict['acc_decision_batch_hv11']))
    message_hv12 = ' acc_decision_batch_hv12: %.4f\n' % (
        np.mean(acc_decision_batch_dict['acc_decision_batch_hv12']))
    message_soft = ' soft_result_acc_list: %.4f\n' % (
        np.mean(soft_result_acc_list))
    message_soft_delete_number = ' soft_delete_number: %d\n' % \
                                 acc_decision_batch_dict['soft_delete_number_total']

    print_and_save_txt(str=message_hv2 + message_hv3 + message_hv4 + message_hv5
                        + message_hv6 + message_hv7 + message_hv8 + message_hv9
                        + message_hv10 + message_hv11 + message_hv12 + message_soft
                        + message_soft_delete_number,
                       filename=os.path.join(backup_path, 'test_log.txt'))


    print_and_save_txt('aborted_num_hv2 {} aborted_num_hv3 {} aborted_num_hv4 {}\n'
                       'aborted_num_hv5 {} aborted_num_hv6 {} aborted_num_hv7 {}\n'
                       'aborted_num_hv8 {} aborted_num_hv9 {} aborted_num_hv10 {}\n'
                       'aborted_num_hv11 {} aborted_num_hv12 {} \n'
                       .format(acc_decision_batch_dict['aborted_num_hv2'],
                               acc_decision_batch_dict['aborted_num_hv3'],
                               acc_decision_batch_dict['aborted_num_hv4'],
                               acc_decision_batch_dict['aborted_num_hv5'],
                               acc_decision_batch_dict['aborted_num_hv6'],
                               acc_decision_batch_dict['aborted_num_hv7'],
                               acc_decision_batch_dict['aborted_num_hv8'],
                               acc_decision_batch_dict['aborted_num_hv9'],
                               acc_decision_batch_dict['aborted_num_hv10'],
                               acc_decision_batch_dict['aborted_num_hv11'],
                               acc_decision_batch_dict['aborted_num_hv12'],

                               )
                       , filename=os.path.join(backup_path, 'test_log.txt'))

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
    for sess in sess_list:
        sess.close()


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


class ConvNet_4_kinds():
    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=44, is_training=True, setting=None):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers
        self.is_training = is_training
        self.setting = setting

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
            stride=1, activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv1')

        dense_layer_1 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_1')

        # dense_layer_2 = DenseLayer(
        #     input_shape=(None, 490),
        #     hidden_dim=self.n_classes,
        #     activation='none', dropout=False, keep_prob=None,
        #     batch_normal=False, weight_decay=1e-4, name='dense_layer_2')

        dense_layer_3 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_3')

        # dense_layer_3_3 = DenseLayer(
        #     input_shape=(None, 512),
        #     hidden_dim=self.n_classes,
        #     activation='none', dropout=False, keep_prob=None,
        #     batch_normal=False, weight_decay=1e-4, name='dense_layer_3_3')

        # 数据流
        # 父层
        basic_conv = conv_layer1.get_output(input=self.images)

        # 子层1
        hidden_conv_1 = self.residual_inference(images=basic_conv, scope_name='son_1_1')
        hidden_conv_1 = self.son_google_v2_part(hidden_conv_1, 'son_1_2')
        input_dense_1 = tf.reduce_mean(hidden_conv_1, reduction_indices=[1, 2])
        self.logits_1 = dense_layer_1.get_output(input=input_dense_1)

        # 子层1--孙1 densenet
        self.logits_1_1 = self.grand_son_1_1(hidden_conv_1, scope_name='grand_son_1_1')

        # 子层1--孙2 densenet
        self.logits_1_2 = self.grand_son_1_2(hidden_conv_1, scope_name='grand_son_1_2')

        # 子层1--孙3 densenet
        self.logits_1_3 = self.grand_son_1_3(hidden_conv_1, scope_name='grand_son_1_3')

        # 子层2
        hidden_conv_2 = self.residual_inference(images=basic_conv, scope_name='son_2_1')
        hidden_conv_2, self.logits_2 = self.son_google_v3_part_2(hidden_conv_2, 'son_2_2')

        # input_dense_2 = tf.reduce_mean(hidden_conv_2, reduction_indices=[1, 2])
        # self.logits_2 = dense_layer_2.get_output(input=input_dense_2)

        # 子层2--孙1   -- densenet_bc
        self.logits_2_1 = self.grand_son_2_1(hidden_conv_2, scope_name='grand_son_2_1')

        # 子层2--孙2   -- densenet_bc
        self.logits_2_2 = self.grand_son_2_2(hidden_conv_2, scope_name='grand_son_2_2')

        # 子层2--孙3   -- densenet_bc
        self.logits_2_3 = self.grand_son_2_3(hidden_conv_2, scope_name='grand_son_2_3')

        # 子层3
        self.is_son3 = False
        hidden_conv_3 = self.residual_inference(images=basic_conv, scope_name='son_3')
        # hidden_conv_3 =self.son_cnn_part_3(hidden_conv_3,scope_name='net_3_2')
        input_dense_3 = tf.reduce_mean(hidden_conv_3, reduction_indices=[1, 2])
        self.logits_3 = dense_layer_3.get_output(input=input_dense_3)

        # 子层3--孙1   Shuffle net
        self.logits_3_1 = self.grand_son_3_1(hidden_conv_3, scope_name='grand_son_3_1')

        # 子层3--孙2   dense bc
        self.logits_3_2 = self.grand_son_3_2(hidden_conv_3, scope_name='grand_son_3_2')

        # 子层3--孙3   Inception reduction b
        self.logits_3_3 = self.grand_son_3_3(hidden_conv_3, scope_name='grand_son_3_3')

        # self.logits_3_3 = dense_layer_3_3.get_output(input=input_dense_3_3)
        # 目标函数
        logits_list = [
            self.logits_1, self.logits_1_1, self.logits_1_2, self.logits_1_3,
            self.logits_2, self.logits_2_1, self.logits_2_2, self.logits_2_3,
            self.logits_3, self.logits_3_1, self.logits_3_2, self.logits_3_3,
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
        lr = tf.cond(tf.less(self.global_step, 30000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 60000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 80000),
                                                     lambda: tf.constant(0.001),
                                                     lambda: tf.constant(0.0005))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.train_op = slim.learning.create_train_op(self.avg_loss, self.optimizer)

        # 精度观察
        # son1
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

        # 观察值son3的 grand son1
        correct_prediction_3_1 = tf.equal(self.labels, tf.argmax(self.logits_3_1, 1))
        self.accuracy_3_1 = tf.reduce_mean(tf.cast(correct_prediction_3_1, 'float'))

        # 观察值son3的 grand son2
        correct_prediction_3_2 = tf.equal(self.labels, tf.argmax(self.logits_3_2, 1))
        self.accuracy_3_2 = tf.reduce_mean(tf.cast(correct_prediction_3_2, 'float'))

        # 观察值son3的 grand son3
        correct_prediction_3_3 = tf.equal(self.labels, tf.argmax(self.logits_3_3, 1))
        self.accuracy_3_3 = tf.reduce_mean(tf.cast(correct_prediction_3_3, 'float'))

        # 概率观察
        self.probability_op = {}
        self.probability_op['logits_1'] = tf.nn.softmax(self.logits_1)
        self.probability_op['logits_1_1'] = tf.nn.softmax(self.logits_1_1)
        self.probability_op['logits_1_2'] = tf.nn.softmax(self.logits_1_2)
        self.probability_op['logits_1_3'] = tf.nn.softmax(self.logits_1_3)
        self.probability_op['logits_2'] = tf.nn.softmax(self.logits_2)
        self.probability_op['logits_2_1'] = tf.nn.softmax(self.logits_2_1)
        self.probability_op['logits_2_2'] = tf.nn.softmax(self.logits_2_2)
        self.probability_op['logits_2_3'] = tf.nn.softmax(self.logits_2_3)
        self.probability_op['logits_3'] = tf.nn.softmax(self.logits_3)
        self.probability_op['logits_3_1'] = tf.nn.softmax(self.logits_3_1)
        self.probability_op['logits_3_2'] = tf.nn.softmax(self.logits_3_2)
        self.probability_op['logits_3_3'] = tf.nn.softmax(self.logits_3_3)

    def grand_son_1_1(self, hidden_conv_1_1, scope_name):
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
                    dropout_keep_prob = 1
                    is_training = self.is_training
                    # is_training= True

                    with tf.variable_scope("block_3"):
                        hidden_conv_1_1 = slim.repeat(hidden_conv_1_1, 4, self.add_internal_layer,
                                                      300, is_training, bc_mode, dropout_keep_prob)

                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_1, self.n_classes, is_training)
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
                    bc_mode = False
                    dropout_keep_prob = 1
                    is_training = self.is_training
                    # is_training= True

                    with tf.variable_scope("block_3"):
                        hidden_conv_1_2 = slim.repeat(hidden_conv_1_2, 4, self.add_internal_layer,
                                                      300, is_training, bc_mode, dropout_keep_prob)

                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_2, self.n_classes, is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def grand_son_1_3(self, hidden_conv_1_3, scope_name):
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
                    dropout_keep_prob = 1
                    is_training = self.is_training
                    # is_training= True

                    with tf.variable_scope("block_3"):
                        hidden_conv_1_3 = slim.repeat(hidden_conv_1_3, 4, self.add_internal_layer,
                                                      300, is_training, bc_mode, dropout_keep_prob)

                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_3, self.n_classes, is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def grand_son_2_1(self, hidden_conv_2_1, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):
                    bc_mode = True
                    dropout_keep_prob = 0.9
                    with tf.variable_scope("block_3"):
                        hidden_conv_2_1 = slim.repeat(hidden_conv_2_1, 3, self.add_internal_layer,
                                                      400, self.is_training, True, 0.9)
                        hidden_conv_2_1 = self.transition_layer(hidden_conv_2_1, 600,
                                                                self.is_training, dropout_keep_prob, 0.9)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_2_1, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
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
                    bc_mode = True
                    dropout_keep_prob = 0.9
                    with tf.variable_scope("block_3"):
                        hidden_conv_2_2 = slim.repeat(hidden_conv_2_2, 3, self.add_internal_layer,
                                                      400, self.is_training, True, 0.9)
                        hidden_conv_2_2 = self.transition_layer(hidden_conv_2_2, 600,
                                                                self.is_training, dropout_keep_prob, 0.9)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_2_2, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def grand_son_2_3(self, hidden_conv_2_3, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):
                    bc_mode = True
                    dropout_keep_prob = 0.9
                    with tf.variable_scope("block_3"):
                        hidden_conv_2_3 = slim.repeat(hidden_conv_2_3, 3, self.add_internal_layer,
                                                      400, self.is_training, True, 0.9)
                        hidden_conv_2_3 = self.transition_layer(hidden_conv_2_3, 600,
                                                                self.is_training, dropout_keep_prob, 0.9)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_2_3, self.n_classes,
                                                                       self.is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def son_google_v3_part_2(self, hidden_conv, scope_name):
        net1 = hidden_conv
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    net1 = slim.repeat(net1, 10, self.block8, scale=0.2)
                    net2 = slim.avg_pool2d(net1, net1.get_shape()[1:3], padding='VALID',
                                           scope='AvgPool_1a_8x8')
                    net2 = slim.flatten(net2)
                    self.logits_2 = slim.fully_connected(net2, self.n_classes, activation_fn=None, scope='Logits')

                    return net1, self.logits_2

        # # google_v2_part InceptionV3
        # with tf.variable_scope(scope_name):
        #     with slim.arg_scope(
        #             [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        #             stride=1,
        #             padding='SAME'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(hidden_conv, depth(80), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(hidden_conv, depth(90), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_1, depth(90), [1, 3], scope='Conv2d_0b_1x3'),
        #                 slim.conv2d(branch_1, depth(90), [3, 1], scope='Conv2d_0c_3x1')])
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(hidden_conv, depth(120), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(
        #                 branch_2, depth(90), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_2, depth(90), [1, 3], scope='Conv2d_0c_1x3'),
        #                 slim.conv2d(branch_2, depth(90), [3, 1], scope='Conv2d_0d_3x1')])
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(hidden_conv, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(
        #                 branch_3, depth(50), [1, 1], scope='Conv2d_0b_1x1')
        #             hidden_conv = tf.concat(
        #                 axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        # return hidden_conv

    def block35(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Builds the 35x35 resnet block."""
        with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                             activation_fn=None, scope='Conv2d_1x1')
            scaled_up = up * scale
            if activation_fn == tf.nn.relu6:
                # Use clip_by_value to simulate bandpass activation.
                scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

            net += scaled_up
            if activation_fn:
                net = activation_fn(net)
        return net

    def inception_resnet_v2_arg_scope(self, weight_decay=0.00004,
                                      batch_norm_decay=0.9997,
                                      batch_norm_epsilon=0.001,
                                      activation_fn=tf.nn.relu):
        """Returns the scope with the default parameters for inception_resnet_v2.

        Args:
          weight_decay: the weight decay for weights variables.
          batch_norm_decay: decay for the moving average of batch_norm momentums.
          batch_norm_epsilon: small float added to variance to avoid dividing by zero.
          activation_fn: Activation function for conv2d.

        Returns:
          a arg_scope with the parameters needed for inception_resnet_v2.
        """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
                'fused': None,  # Use fused batch norm if possible.
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params) as scope:
                return scope

    def son_google_v2_part(self, hidden_conv, scope_name):
        net = hidden_conv
        activation_fn = tf.nn.relu
        # google_v2_part
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                net = slim.repeat(net, 10, self.block35, scale=0.17,
                                  activation_fn=activation_fn)

        return net

    def transition_layer(self, _input, num_filter, training=True, dropout_keep_prob=0.8, reduction=1.0):
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

    def trainsition_layer_to_classes(self, _input, n_classes=10, training=True):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        _output = slim.batch_norm(_input)
        # L.scale in caffe
        _output = tf.nn.relu(_output)
        last_pool_kernel = int(_output.get_shape()[-2])
        _output = slim.avg_pool2d(_output, [last_pool_kernel, last_pool_kernel])
        logits = slim.fully_connected(_output, n_classes)
        return logits

    def composite_function(self, _input, out_features, training=True, dropout_keep_prob=0.8, kernel_size=[3, 3]):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = slim.batch_norm(_input)  # !!need op
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = slim.conv2d(output, out_features, kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            if training:
                output = slim.dropout(output, dropout_keep_prob)
        return output

    def bottleneck(self, _input, out_features, training=True, dropout_keep_prob=0.8):
        with tf.variable_scope("bottleneck"):
            inter_features = out_features * 4
            output = slim.batch_norm(_input)  # !!need op
            ################################################!!!!!!
            output = tf.nn.relu(output)
            output = slim.conv2d(output, inter_features, [1, 1], padding='VALID')
            if training:
                output = slim.dropout(output, dropout_keep_prob)
        return output

    def add_internal_layer(self, _input, growth_rate, training=True, bc_mode=False, dropout_keep_prob=1.0,
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

    def grand_son_3_1(self, net, scope_name):

        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    net = slim.repeat(net, 9, self.block8, scale=0.2)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    logits = slim.fully_connected(net, self.n_classes, activation_fn=None,
                                                  scope='Logits')
                    return logits

    def grand_son_3_2(self, net, scope_name):

        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    net = slim.repeat(net, 9, self.block8, scale=0.2)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    logits = slim.fully_connected(net, self.n_classes, activation_fn=None,
                                                  scope='Logits')
                    return logits

    def block8(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """构建8x8的resnet块."""
        with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                            scope='Conv2d_0b_1x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                            scope='Conv2d_0c_3x1')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                             activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net

    def grand_son_3_3(self, net, scope_name):

        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    net = slim.repeat(net, 9, self.block8, scale=0.2)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    logits = slim.fully_connected(net, self.n_classes, activation_fn=None,
                                                  scope='Logits')
                    return logits

    def residual_inference(self, images, scope_name):

        with tf.variable_scope(scope_name):
            n_layers = int((self.n_layers - 2) / 6)
            # if self.is_son3:
            #     n_layers=44
            #     self.is_son3 =False
            # 网络结构
            conv_layer0_list = []
            conv_layer0_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 3),
                    n_size=3, n_filter=64, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv0'))

            conv_layer1_list = []
            for i in range(1, n_layers + 1):
                conv_layer1_list.append(
                    ConvLayer(
                        input_shape=(None, self.image_size, self.image_size, 64),
                        n_size=3, n_filter=64, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2 * i - 1)))
                conv_layer1_list.append(
                    ConvLayer(
                        input_shape=(None, self.image_size, self.image_size, 64),
                        n_size=3, n_filter=64, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv1_%d' % (2 * i)))

            conv_layer2_list = []
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, self.image_size, self.image_size, 64),
                    n_size=3, n_filter=128, stride=2, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv2_1'))
            conv_layer2_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size) / 2, int(self.image_size) / 2, 128),
                    n_size=3, n_filter=128, stride=1, activation='none',
                    batch_normal=True, weight_decay=1e-4, name='conv2_2'))
            for i in range(2, n_layers + 1):
                conv_layer2_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size / 2), int(self.image_size / 2), 128),
                        n_size=3, n_filter=128, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2 * i - 1)))
                conv_layer2_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size / 2), int(self.image_size / 2), 128),
                        n_size=3, n_filter=128, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv2_%d' % (2 * i)))

            conv_layer3_list = []
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size / 2), int(self.image_size / 2), 128),
                    n_size=3, n_filter=256, stride=2, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv3_1'))
            conv_layer3_list.append(
                ConvLayer(
                    input_shape=(None, int(self.image_size / 4), int(self.image_size / 4), 256),
                    n_size=3, n_filter=256, stride=1, activation='relu',
                    batch_normal=True, weight_decay=1e-4, name='conv3_2'))
            for i in range(2, n_layers + 1):
                conv_layer3_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size / 4), int(self.image_size / 4), 256),
                        n_size=3, n_filter=256, stride=1, activation='relu',
                        batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2 * i - 1)))
                conv_layer3_list.append(
                    ConvLayer(
                        input_shape=(None, int(self.image_size / 4), int(self.image_size / 4), 256),
                        n_size=3, n_filter=256, stride=1, activation='none',
                        batch_normal=True, weight_decay=1e-4, name='conv3_%d' % (2 * i)))

            # 数据流
            hidden_conv = conv_layer0_list[0].get_output(input=images)

            for i in range(0, n_layers):
                hidden_conv1 = conv_layer1_list[2 * i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer1_list[2 * i + 1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

            hidden_conv1 = conv_layer2_list[0].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer2_list[1].get_output(input=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0, 0], [0, 0], [0, 0], [32, 32]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for i in range(1, n_layers):
                hidden_conv1 = conv_layer2_list[2 * i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer2_list[2 * i + 1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

            hidden_conv1 = conv_layer3_list[0].get_output(input=hidden_conv)
            hidden_conv2 = conv_layer3_list[1].get_output(input=hidden_conv1)
            hidden_pool = tf.nn.max_pool(
                hidden_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            hidden_pad = tf.pad(hidden_pool, [[0, 0], [0, 0], [0, 0], [64, 64]])
            hidden_conv = tf.nn.relu(hidden_pad + hidden_conv2)
            for i in range(1, n_layers):
                hidden_conv1 = conv_layer3_list[2 * i].get_output(input=hidden_conv)
                hidden_conv2 = conv_layer3_list[2 * i + 1].get_output(input=hidden_conv1)
                hidden_conv = tf.nn.relu(hidden_conv + hidden_conv2)

            return hidden_conv

class ConvNet_6_kinds():
    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=44,is_training =True,setting=None):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers
        self.is_training  = is_training
        self.setting = setting

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
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_1')

        # dense_layer_2 = DenseLayer(
        #     input_shape=(None, 490),
        #     hidden_dim=self.n_classes,
        #     activation='none', dropout=False, keep_prob=None,
        #     batch_normal=False, weight_decay=1e-4, name='dense_layer_2')

        dense_layer_3 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense_layer_3')

        # dense_layer_3_3 = DenseLayer(
        #     input_shape=(None, 512),
        #     hidden_dim=self.n_classes,
        #     activation='none', dropout=False, keep_prob=None,
        #     batch_normal=False, weight_decay=1e-4, name='dense_layer_3_3')

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
        hidden_conv_2, self.logits_2 = self.son_google_v3_part_2(hidden_conv_2,'son_2_2')

        # input_dense_2 = tf.reduce_mean(hidden_conv_2, reduction_indices=[1, 2])
        # self.logits_2 = dense_layer_2.get_output(input=input_dense_2)


        # 子层2--孙1   -- depth conv L8
        self.logits_2_1 = self.grand_son_2_1(hidden_conv_2, scope_name='grand_son_2_1')

        # 子层2--孙2   -- densenet_bc
        self.logits_2_2 = self.grand_son_2_2(hidden_conv_2, scope_name='grand_son_2_2')

        # 子层2--孙3   -- shuffle
        self.logits_2_3 = self.grand_son_2_3(hidden_conv_2, scope_name='grand_son_2_3')


        # 子层3
        self.is_son3 =False
        hidden_conv_3 = self.residual_inference(images=basic_conv, scope_name='son_3')
        # hidden_conv_3 =self.son_cnn_part_3(hidden_conv_3,scope_name='net_3_2')
        input_dense_3= tf.reduce_mean(hidden_conv_3, reduction_indices=[1, 2])
        self.logits_3 = dense_layer_3.get_output(input=input_dense_3)

        # 子层3--孙1   Shuffle net
        self.logits_3_1 = self.grand_son_3_1(hidden_conv_3, scope_name='grand_son_3_1')

        # 子层3--孙2   dense bc
        self.logits_3_2 = self.grand_son_3_2(hidden_conv_3, scope_name='grand_son_3_2')

        # 子层3--孙3   Inception reduction b
        self.logits_3_3 = self.grand_son_3_3(hidden_conv_3, scope_name='grand_son_3_3')

        # self.logits_3_3 = dense_layer_3_3.get_output(input=input_dense_3_3)
        # 目标函数
        logits_list=[
                    self.logits_1,self.logits_1_1,self.logits_1_2,self.logits_1_3,
                    self.logits_2,self.logits_2_1,self.logits_2_2,self.logits_2_3,
                    self.logits_3,self.logits_3_1,self.logits_3_2,self.logits_3_3,
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
        lr = tf.cond(tf.less(self.global_step, 30000),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 60000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 80000),
                                                     lambda: tf.constant(0.001),
                                                     lambda: tf.constant(0.0005))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.train_op = slim.learning.create_train_op(self.avg_loss, self.optimizer)

        # 精度观察
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

        # 观察值son3的 grand son1
        correct_prediction_3_1 = tf.equal(self.labels, tf.argmax(self.logits_3_1, 1))
        self.accuracy_3_1 = tf.reduce_mean(tf.cast(correct_prediction_3_1, 'float'))

        # 观察值son3的 grand son2
        correct_prediction_3_2 = tf.equal(self.labels, tf.argmax(self.logits_3_2, 1))
        self.accuracy_3_2 = tf.reduce_mean(tf.cast(correct_prediction_3_2, 'float'))
        
        # 观察值son3的 grand son3
        correct_prediction_3_3 = tf.equal(self.labels, tf.argmax(self.logits_3_3, 1))
        self.accuracy_3_3 = tf.reduce_mean(tf.cast(correct_prediction_3_3, 'float'))

        #概率观察
        self.probability_op={}
        self.probability_op['logits_1'] = tf.nn.softmax(self.logits_1)
        self.probability_op['logits_1_1'] = tf.nn.softmax(self.logits_1_1)
        self.probability_op['logits_1_2'] = tf.nn.softmax(self.logits_1_2)
        self.probability_op['logits_1_3'] = tf.nn.softmax(self.logits_1_3)
        self.probability_op['logits_2'] = tf.nn.softmax(self.logits_2)
        self.probability_op['logits_2_1'] = tf.nn.softmax(self.logits_2_1)
        self.probability_op['logits_2_2'] = tf.nn.softmax(self.logits_2_2)
        self.probability_op['logits_2_3'] = tf.nn.softmax(self.logits_2_3)
        self.probability_op['logits_3'] = tf.nn.softmax(self.logits_3)
        self.probability_op['logits_3_1'] = tf.nn.softmax(self.logits_3_1)
        self.probability_op['logits_3_2'] = tf.nn.softmax(self.logits_3_2)
        self.probability_op['logits_3_3'] = tf.nn.softmax(self.logits_3_3)

    def grand_son_1_1(self, hidden_conv_1, scope_name):
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
                    dropout_keep_prob = 1
                    is_training = self.is_training
                    # is_training= True
                    hidden_conv_1_1 = hidden_conv_1
                    with tf.variable_scope("block_3"):
                        hidden_conv_1_1 = slim.repeat(hidden_conv_1_1, 4, self.add_internal_layer,
                                                      300, is_training, bc_mode, dropout_keep_prob)

                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_1, self.n_classes, is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def grand_son_arg_scope_1_2(self,is_training=True,
                                weight_decay=0.00004,
                                stddev=0.09,
                                regularize_depthwise=False,
                                batch_norm_decay=0.99,
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

                    is_training =self.is_training
                    with tf.variable_scope("block_3"):
                        hidden_conv_1_2 = slim.repeat(hidden_conv_1_2, 4, self.add_internal_layer,
                                                      300, is_training,  True, 0.8)
                        hidden_conv_1_2 = self.transition_layer(hidden_conv_1_2, 500, is_training)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_1_2, self.n_classes,
                                                                       is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits



    def grand_son_1_3(self, net, scope_name):
        global_pool = True
        padding = 'SAME'
        use_explicit_padding = True
        spatial_squeeze = True
        dropout_keep_prob = 0.9
        final_endpoint = 'Conv2d_3'
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.grand_son_arg_scope_1_2(is_training=True)):
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
                        if self.is_training:
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
            with slim.arg_scope(self.grand_son_arg_scope_1_2(is_training= True)):
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
                        if self.is_training:
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
                    dropout_keep_prob = 0.9
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
                    x_ret = slim.conv2d(x_ret, inputs.get_shape()[3], 1, normalizer_fn=None,
                                     activation_fn=None, scope='Conv2d_1x1')
                    x_ret = x_ret + inputs
            return slim.utils.collect_named_outputs(outputs_collections, sc.name, x_ret)

    def grand_son_2_3(self, hidden_conv_2_3, scope_name):
        with tf.variable_scope(scope_name) as sc:
            with slim.arg_scope(self.shufflenet_arg_scope(is_training=True)):
                net = self.shuffle_stage(hidden_conv_2_3, depth=256, groups=4, repeat=3, shuffle=True, scope='Stage1')
                net = self.shuffle_stage_v2(net, depth=256 *2, groups=4, repeat=3, shuffle=True, scope='Stage2')

                ##############################
                with tf.variable_scope('Logits'):
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

                    # 1 x 1 x 512
                    if self.is_training:
                        net = slim.dropout(net, keep_prob=0.9, scope='Dropout_1b')
                    # logits = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                    #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    net = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope="conv_1x1")
                    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

                return logits

    def shufflenet_arg_scope(self,is_training=True,
                             weight_decay=0.0001,
                             batch_norm_decay=0.997,
                             batch_norm_epsilon=1e-5,
                             batch_norm_scale=True):

        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def channel_shuffle_v1(self,input, depth_bottleneck, group, scope=None):
        assert 0 == depth_bottleneck % group, "Output channels must be a multiple of groups"
        with tf.variable_scope(scope, "ChannelShuffle", [input]):
            n, h, w, c = input.shape.as_list()
            x_reshape = tf.reshape(input, [-1, h, w, group, depth_bottleneck // group])
            x_transposed = tf.transpose(x_reshape, [0, 1, 2, 4, 3])
            net = tf.reshape(x_transposed, [-1, h, w, c])
            return net

    def group_pointwise_conv2d(self,inputs, depth, stride, group, relu=True, scope=None):

        assert 0 == depth % group, "Output channels must be a multiple of groups"
        num_channels_in_group = depth // group
        with tf.variable_scope(scope, 'GConv', [inputs]) as sc:
            net = tf.split(inputs, group, axis=3, name="split")
            for i in range(group):
                net[i] = slim.conv2d(net[i],
                                     num_channels_in_group,
                                     [1, 1],
                                     stride=stride,
                                     activation_fn=None,
                                     normalizer_fn=None)
            net = tf.concat(net, axis=3, name="concat")
            net = slim.batch_norm(net, activation_fn=tf.nn.relu if relu else None)
        return net

    @slim.add_arg_scope
    def shuffle_bottleneck(self,inputs, depth_bottleneck, group, stride, shuffle=True, outputs_collections=None, scope=None):
        if 1 != stride:
            _b, _h, _w, _c = inputs.get_shape().as_list()
            depth_bottleneck = depth_bottleneck - _c

        assert 0 == depth_bottleneck % group, "Output channels must be a multiple of groups"

        with tf.variable_scope(scope, 'Unit', [inputs]) as sc:
            print("shuffle_bottleneck", sc.name)
            if 1 != stride:
                net_skip = slim.avg_pool2d(inputs, [3, 3], stride, padding="SAME", scope='3x3AVGPool2D')
            else:
                net_skip = inputs

            net = self.group_pointwise_conv2d(inputs, depth_bottleneck, 1,
                                         group= group,
                                         relu=True, scope="1x1GConvIn")

            if shuffle:
                net = self.channel_shuffle_v1(net, depth_bottleneck, group)

            with tf.variable_scope("3x3DWConv"):
                depthwise_filter = tf.get_variable("depth_conv_w", [3, 3, depth_bottleneck, 1],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.01))
                net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")
                # Todo: Add batch norm here
                net = slim.batch_norm(net, activation_fn=None)

            net = self.group_pointwise_conv2d(net, depth_bottleneck, 1, group, relu=False, scope="1x1GConvOut")

            if 1 != stride:
                net = tf.concat([net, net_skip], axis=3)
            else:
                net = net + net_skip
            out = tf.nn.relu(net)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, out)

    def shuffle_stage(self,inputs, depth, groups, repeat, shuffle=True, scope=None):
        with tf.variable_scope(scope, "Stage", [inputs]) as sc:
            for i in range(repeat):
                net = self.shuffle_bottleneck(inputs, depth, group=groups, stride=1, shuffle=shuffle,
                                         scope='Unit{}'.format(i + 1))
            return net
    def shuffle_stage_v2(self,inputs, depth, groups, repeat, shuffle=True, scope=None):
        with tf.variable_scope(scope, "Stage", [inputs]) as sc:
            net = self.shuffle_bottleneck(inputs, depth, group=groups, stride=2, shuffle=shuffle,
                                      scope='Unit_start_first')
            for i in range(repeat):
                net = self.shuffle_bottleneck(net, depth, group=groups, stride=1, shuffle=shuffle,
                                         scope='Unit{}'.format(i + 1))
            return net

    def grand_son_3_1(self, hidden_conv_3_1, scope_name):
        with tf.variable_scope(scope_name) as sc:
            with slim.arg_scope(self.shufflenet_arg_scope(is_training=True)):
                net =self.shuffle_stage(hidden_conv_3_1, depth=256, groups=4, repeat=3, shuffle=True, scope=None)

                ##############################
                with tf.variable_scope('Logits'):

                        # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')

                    # 1 x 1 x 512
                    if self.is_training:
                        net = slim.dropout(net, keep_prob=0.9, scope='Dropout_1b')
                    # logits = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                    #                      normalizer_fn=None, scope='Conv2d_1c_1x1')
                    net = slim.conv2d(net, self.n_classes, [1, 1], activation_fn=None,
                                         normalizer_fn=None, scope="conv_1x1")
                    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

                return logits
                ###########################




    def son_cnn_part_3(self, hidden_conv, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
                hidden_conv = slim.conv2d(hidden_conv, depth(200), [3, 3], scope='Conv2d_1')
        return hidden_conv

    def son_google_v3_part_2(self, hidden_conv, scope_name):
        net1 = hidden_conv
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    net1 = slim.repeat(net1, 10, self.block8, scale=0.2)
                    net2 = slim.avg_pool2d(net1, net1.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net2 = slim.flatten(net2)
                    self.logits_2 = slim.fully_connected(net2, self.n_classes, activation_fn=None, scope='Logits')

                    return net1,self.logits_2

        # # google_v2_part InceptionV3
        # with tf.variable_scope(scope_name):
        #     with slim.arg_scope(
        #             [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        #             stride=1,
        #             padding='SAME'):
        #         with tf.variable_scope('Branch_0'):
        #             branch_0 = slim.conv2d(hidden_conv, depth(80), [1, 1], scope='Conv2d_0a_1x1')
        #         with tf.variable_scope('Branch_1'):
        #             branch_1 = slim.conv2d(hidden_conv, depth(90), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_1 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_1, depth(90), [1, 3], scope='Conv2d_0b_1x3'),
        #                 slim.conv2d(branch_1, depth(90), [3, 1], scope='Conv2d_0c_3x1')])
        #         with tf.variable_scope('Branch_2'):
        #             branch_2 = slim.conv2d(hidden_conv, depth(120), [1, 1], scope='Conv2d_0a_1x1')
        #             branch_2 = slim.conv2d(
        #                 branch_2, depth(90), [3, 3], scope='Conv2d_0b_3x3')
        #             branch_2 = tf.concat(axis=3, values=[
        #                 slim.conv2d(branch_2, depth(90), [1, 3], scope='Conv2d_0c_1x3'),
        #                 slim.conv2d(branch_2, depth(90), [3, 1], scope='Conv2d_0d_3x1')])
        #         with tf.variable_scope('Branch_3'):
        #             branch_3 = slim.avg_pool2d(hidden_conv, [3, 3], scope='AvgPool_0a_3x3')
        #             branch_3 = slim.conv2d(
        #                 branch_3, depth(50), [1, 1], scope='Conv2d_0b_1x1')
        #             hidden_conv = tf.concat(
        #                 axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        # return hidden_conv

    def block35(self,net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """Builds the 35x35 resnet block."""
        with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
                tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
                tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
                tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                             activation_fn=None, scope='Conv2d_1x1')
            scaled_up = up * scale
            if activation_fn == tf.nn.relu6:
                # Use clip_by_value to simulate bandpass activation.
                scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

            net += scaled_up
            if activation_fn:
                net = activation_fn(net)
        return net

    def inception_resnet_v2_arg_scope(self,weight_decay=0.00004,
                                      batch_norm_decay=0.9997,
                                      batch_norm_epsilon=0.001,
                                      activation_fn=tf.nn.relu):
        """Returns the scope with the default parameters for inception_resnet_v2.

        Args:
          weight_decay: the weight decay for weights variables.
          batch_norm_decay: decay for the moving average of batch_norm momentums.
          batch_norm_epsilon: small float added to variance to avoid dividing by zero.
          activation_fn: Activation function for conv2d.

        Returns:
          a arg_scope with the parameters needed for inception_resnet_v2.
        """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_regularizer=slim.l2_regularizer(weight_decay)):
            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
                'fused': None,  # Use fused batch norm if possible.
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params) as scope:
                return scope

    def son_google_v2_part(self,hidden_conv,scope_name):
        net = hidden_conv
        activation_fn = tf.nn.relu
        # google_v2_part
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                net = slim.repeat(net, 10, self.block35, scale=0.17,
                                  activation_fn=activation_fn)

        return net

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
        _output = slim.batch_norm(_input)
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
            output = slim.batch_norm(_input)  # !!need op
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
            output = slim.batch_norm(_input)  # !!need op
            ################################################!!!!!!
            output = tf.nn.relu(output)
            output = slim.conv2d(output, inter_features, [1, 1], padding='VALID')
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

    def grand_son_3_2(self, hidden_conv_3_2, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=None,
                                normalizer_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0004),
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                biases_initializer=tf.zeros_initializer()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME'):

                    is_training =self.is_training
                    with tf.variable_scope("block_3"):
                        hidden_conv_3_2 = slim.repeat(hidden_conv_3_2, 3, self.add_internal_layer,
                                                      300, is_training,  True, 0.8)
                        hidden_conv_3_2 = self.transition_layer(hidden_conv_3_2, 500, is_training)
                        with tf.variable_scope("trainsition_layer_to_classes"):
                            logits = self.trainsition_layer_to_classes(hidden_conv_3_2, self.n_classes,
                                                                       is_training)
                            logits = tf.reshape(logits, [-1, self.n_classes])
        return logits

    def block_inception_b(self,inputs, scope=None, reuse=None):
        """Builds Inception-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 112, [1, 3], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 64, [3, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs, 86, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 86, [3, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 112, [1, 3], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 112, [3, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 3], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def block8(self,net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
        """构建8x8的resnet块."""
        with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
            with tf.variable_scope('Branch_0'):
                tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
                tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
                tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                            scope='Conv2d_0b_1x3')
                tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                            scope='Conv2d_0c_3x1')
            mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
            up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                             activation_fn=None, scope='Conv2d_1x1')
            net += scale * up
            if activation_fn:
                net = activation_fn(net)
        return net
    def grand_son_3_3(self, hidden_conv_3_3, scope_name):
        net = hidden_conv_3_3
        with tf.variable_scope(scope_name):
            with slim.arg_scope(self.inception_resnet_v2_arg_scope()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):

                    net = slim.repeat(net, 9, self.block8, scale=0.2)
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
                    logits = slim.fully_connected(net, self.n_classes, activation_fn=None,
                                                  scope='Logits')
                    return logits


    def residual_inference(self, images,scope_name):

        with tf.variable_scope(scope_name):
            n_layers = int((self.n_layers - 2) / 6)
            # if self.is_son3:
            #     n_layers=44
            #     self.is_son3 =False
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



        


