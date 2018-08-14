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
        self.logits = self.inference(self.images)
        
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
        
    def inference(self, images):
        n_layers = int((self.n_layers - 2) / 6)
        # 网络结构
        conv_layer0_list = []
        conv_layer0_list.append(
            ConvLayer(
                input_shape=(None, self.image_size, self.image_size, self.n_channel), 
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
        
        dense_layer1 = DenseLayer(
            input_shape=(None, 256),
            hidden_dim=self.n_classes,
            activation='none', dropout=False, keep_prob=None, 
            batch_normal=False, weight_decay=1e-4, name='dense1')
        
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
            
        # global average pooling
        input_dense1 = tf.reduce_mean(hidden_conv, reduction_indices=[1, 2])
        logits = dense_layer1.get_output(input=input_dense1)
        
        return logits
        
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
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
        valid_labels = dataloader.valid_labels
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 训练集数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
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
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
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