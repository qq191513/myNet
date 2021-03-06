# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer
from src.model.squeezenet import squeezenet
from src.model.squeezenet import squeezenet_arg_scope
slim = tf.contrib.slim

def print_and_save_txt(str=None,filename=r'log.txt'):
    with open(filename, "a+") as log_writter:
        print(str)
        log_writter.write(str)
class ConvNet():
    def __init__(self, n_channel=3, n_classes=10, image_size=24, n_layers=20):
        # 设置超参数
        self.n_channel = n_channel
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_layers = n_layers
        
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

        conv_layer2 = ConvLayer(
            input_shape=(None, image_size, image_size, 3), n_size=3, n_filter=3,
            stride=1,activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv2')

        # 数据流
        basic_conv_1 = conv_layer1.get_output(input=self.images)
        basic_conv_2 = conv_layer2.get_output(input=basic_conv_1)
        with slim.arg_scope(squeezenet_arg_scope()):
            self.logits,end_points = squeezenet(basic_conv_2, 10,scope='SqueezeNet_v1')


        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000), 
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                     lambda: tf.constant(0.005),
                                     lambda: tf.cond(tf.less(self.global_step, 150000),
                                                     lambda: tf.constant(0.0025),
                                                     lambda: tf.constant(0.001))))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        correct_prediction = tf.equal(self.labels, tf.argmax(self.logits, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        

        
    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        log_writter= open(os.path.join(backup_path, 'train_log.txt'), "w")

        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=5)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        
        # 验证集数据增强
        valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
            flip=False, crop=False, crop_shape=(32,32,3), whiten=True, noise=False)
        valid_labels = dataloader.valid_labels
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 训练集数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=False, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            
            # 开始本轮的训练，并计算目标函数值
            train_loss = 0.0
            get_global_step = 0
            for i in range(0, dataloader.n_train, batch_size):
            # for i in range(0, 300, batch_size):
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
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid

            message = 'epoch: %d , global_step: %d , train loss: %.6f , valid precision: %.6f , valid loss: %.6f\n' % (
                epoch, get_global_step, train_loss, valid_accuracy, valid_loss)
            log_writter.write(message)
            print(message)
            sys.stdout.flush()
            

            # 保存模型
            if epoch % 10 == 0 :
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
        log_writter.close()
        self.sess.close()
                
    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        accuracy_list = []
        test_images = dataloader.data_augmentation(dataloader.test_images,
            flip=False, crop=False, crop_shape=(24,24,3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy], 
                feed_dict={self.images:batch_images, 
                           self.labels:batch_labels,
                           self.keep_prob:1.0})
            accuracy_list.append(avg_accuracy)

        print_and_save_txt(str='test precision_net1: %.4f\n' % (np.mean(accuracy_list)),
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

        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print_and_save_txt(str='parament numbers is : %d' % get_num_params(),
                           filename=os.path.join(backup_path, 'test_log.txt'))
        #######################################
        self.sess.close()
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.logits],
            feed_dict={self.images: np.random.random(size=[128, 24, 24, 3]),
                       self.labels: np.random.randint(low=0, high=9, size=[128,]),
                       self.keep_prob: 1.0})
        print(temp.shape)