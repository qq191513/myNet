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
from numpy import *

def net(hidden_pool1,keep_prob,image_size,n_classes,naem_scope):
    with tf.variable_scope(naem_scope):
        conv_layer2_1 = ConvLayer(
            input_shape=(None, int(image_size / 2), int(image_size / 2), 64), n_size=3, n_filter=128,
            stride=1, activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv2')
        pool_layer2_1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool2')

        conv_layer3_1 = ConvLayer(
            input_shape=(None, int(image_size / 4), int(image_size / 4), 128), n_size=3, n_filter=256,
            stride=1, activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv3')
        pool_layer3_1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool3')

        dense_layer1_1 = DenseLayer(
            input_shape=(None, int(image_size / 8) * int(image_size / 8) * 256), hidden_dim=1024,
            activation='relu', dropout=True, keep_prob=keep_prob,
            batch_normal=True, weight_decay=1e-4, name='dense1')

        dense_layer2_1 = DenseLayer(
            input_shape=(None, 1024), hidden_dim=n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense2')

        # net 1
        hidden_conv2_1 = conv_layer2_1.get_output(input=hidden_pool1)
        hidden_pool2_1 = pool_layer2_1.get_output(input=hidden_conv2_1)
        hidden_conv3_1 = conv_layer3_1.get_output(input=hidden_pool2_1)
        hidden_pool3_1 = pool_layer3_1.get_output(input=hidden_conv3_1)
        input_dense1_1 = tf.reshape(hidden_pool3_1, [-1, int(image_size / 8) * int(image_size / 8) * 256])
        output_dense1_1 = dense_layer1_1.get_output(input=input_dense1_1)
        logits = dense_layer2_1.get_output(input=output_dense1_1)
        return logits



class ConvNet():
    
    def __init__(self, n_channel=3, n_classes=10, image_size=24):
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_size, image_size, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable( 
            0, dtype=tf.int32, name='global_step')
        
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(None, image_size, image_size, n_channel), n_size=3, n_filter=64, 
            stride=1,activation='relu', batch_normal=True, weight_decay=1e-4,
            name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool1')
        



        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)


        #net_1
        self.logits_1 = net(hidden_pool1, self.keep_prob, image_size, n_classes, 'net_1')

        #net 2
        self.logits_2 = net(hidden_pool1, self.keep_prob, image_size, n_classes, 'net_2')

        #net 3
        self.logits_3 = net(hidden_pool1, self.keep_prob, image_size, n_classes, 'net_3')


        # 目标函数
        self.objective_1 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_1, labels=self.labels))

        self.objective_2 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_2, labels=self.labels))

        self.objective_3 = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_3, labels=self.labels))


        self.objective = self.objective_1 + self.objective_2 + self.objective_3

        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))

        # 优化器
        lr = tf.cond(tf.less(self.global_step, 50000), 
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100000), 
                                     lambda: tf.constant(0.001),
                                     lambda: tf.constant(0.0001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.avg_loss, global_step=self.global_step)
        
        # 观察值
        correct_prediction_1 = tf.equal(self.labels, tf.argmax(self.logits_1, 1))
        self.accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, 'float'))

        correct_prediction_2 = tf.equal(self.labels, tf.argmax(self.logits_2, 1))
        self.accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, 'float'))

        correct_prediction_3 = tf.equal(self.labels, tf.argmax(self.logits_3, 1))
        self.accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, 'float'))


    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):

        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        log_writter= open(os.path.join(backup_path, 'train_log.txt'), "w")


        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=5)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
                flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            valid_labels = dataloader.valid_labels
            
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
                if get_global_step % 200 == 0:
                    print('global_step {} ,data_batch idx {} , batch_loss :{}'.format(get_global_step,i,avg_loss))
                train_loss += avg_loss * batch_images.shape[0]
            train_loss = 1.0 * train_loss / dataloader.n_train
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            avg_accuracy_1, valid_loss = 0.0, 0.0
            valid_accuracy_1 = 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                # for i in range(0, 300, batch_size):
                batch_images = valid_images[i: i + batch_size]
                batch_labels = valid_labels[i: i + batch_size]
                [avg_accuracy_1, avg_loss] = self.sess.run(
                    fetches=[self.accuracy_1, self.avg_loss],
                    feed_dict={self.images: batch_images,
                               self.labels: batch_labels,
                               self.keep_prob: 1.0})
                valid_accuracy_1 += avg_accuracy_1 * batch_images.shape[0]

                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy_1 = 1.0 * valid_accuracy_1 / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid
            message = 'epoch: %d , global_step: %d , train loss: %.6f , valid precision: %.6f , valid loss: %.6f\n' % (
                epoch, get_global_step, train_loss, valid_accuracy_1, valid_loss)
            log_writter.write(message)
            print(message)
            sys.stdout.flush()
            
            # 保存模型
            if epoch % 5 == 0 :
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
        log_writter.close()
        self.sess.close()
                
    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))

        # 在测试集上计算准确率
        accuracy_1_list = []
        accuracy_2_list = []
        accuracy_3_list = []

        test_images = dataloader.data_augmentation(dataloader.test_images,
            flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        acc_decision_batchs = numpy.float32(.0)
        batchs_number = 0
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]

            [avg_accuracy_1,avg_accuracy_2,avg_accuracy_3,logits_1,logits_2,logits_3] = self.sess.run(
                fetches=[self.accuracy_1,self.accuracy_2,self.accuracy_3,
                         self.logits_1,self.logits_2,self.logits_3],
                feed_dict={self.images:batch_images,
                           self.labels:batch_labels,
                           self.keep_prob:1.0})
            accuracy_1_list.append(avg_accuracy_1)
            accuracy_2_list.append(avg_accuracy_2)
            accuracy_3_list.append(avg_accuracy_3)

            predict_1 = self.sess.run(tf.argmax(logits_1,axis=1))
            predict_2 = self.sess.run(tf.argmax(logits_2,axis=1))
            predict_3 = self.sess.run(tf.argmax(logits_3,axis=1))

            final_batch_predict_list = []
            for idx in range(0,len(predict_1)):
                if(predict_1[idx] == predict_2[idx]):
                    final_batch_predict_list.append(predict_1[idx])
                    continue
                if(predict_1[idx] == predict_3[idx]):
                    final_batch_predict_list.append(predict_1[idx])
                    continue
                if(predict_2[idx] == predict_3[idx]):
                    final_batch_predict_list.append(predict_3[idx])
                    continue
                final_batch_predict_list.append(predict_1[idx])
            array_final_batch_predict_list = array(final_batch_predict_list)

            totol_batch_prediction = tf.equal(self.labels, array_final_batch_predict_list)

            self.decision_batch_prediction = tf.reduce_mean(tf.cast(totol_batch_prediction, 'float'))
            acc_decision_batch = self.sess.run(self.decision_batch_prediction,feed_dict={self.labels:batch_labels})
            acc_decision_batchs =acc_decision_batchs + acc_decision_batch
            batchs_number = batchs_number + 1
            # print('acc_decision_batch:',acc_decision_batch)
                # print('predict_2:',predict_2[idx])
                # print('predict_3:',predict_3[idx])

            # print('predict_1:', predict_1[1])
            # print('predict_2:', predict_2[1])
            # print('predict_3:', predict_3[1])


        mean_acc_decision = acc_decision_batchs / batchs_number
        print('test precision_v1: %.4f' % (numpy.mean(accuracy_1_list)))
        print('test precision_v2: %.4f' % (numpy.mean(accuracy_2_list)))
        print('test precision_v3: %.4f' % (numpy.mean(accuracy_3_list)))
        print('self.decision_prediction: %.4f' % mean_acc_decision )

        #########  parameters numbers###########
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




        self.sess.close()
            
    def debug(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        [temp] = sess.run(
            fetches=[self.observe],
            feed_dict={self.images: numpy.random.random(size=[128, 24, 24, 3]),
                       self.labels: numpy.random.randint(low=0, high=9, size=[128,]),
                       self.keep_prob: 1.0})
        print(temp)
        
    # def observe_salience(self, batch_size=128, image_h=32, image_w=32, n_channel=3,
    #                      num_test=10, epoch=1):
    #     if not os.path.exists('results/epoch%d/' % (epoch)):
    #         os.makedirs('results/epoch%d/' % (epoch))
    #     saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    #     sess = tf.Session()
    #     # 读取模型
    #     model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
    #     assert(os.path.exists(model_path+'.index'))
    #     saver.restore(sess, model_path)
    #     print('read model from %s' % (model_path))
    #     # 获取图像并计算梯度
    #     for batch in range(num_test):
    #         batch_image, batch_label = cifar10.test.next_batch(batch_size)
    #         image = numpy.array(batch_image.reshape([image_h, image_w, n_channel]) * 255,
    #                             dtype='uint8')
    #         result = sess.run([self.labels_prob, self.labels_max_prob, self.labels_pred,
    #                            self.gradient],
    #                           feed_dict={self.images:batch_image, self.labels:batch_label,
    #                                      self.keep_prob:0.5})
    #         print(result[0:3], result[3][0].shape)
    #         gradient = sess.run(self.gradient, feed_dict={
    #             self.images:batch_image, self.keep_prob:0.5})
    #         gradient = gradient[0].reshape([image_h, image_w, n_channel])
    #         gradient = numpy.max(gradient, axis=2)
    #         gradient = numpy.array((gradient - gradient.min()) * 255
    #                                 / (gradient.max() - gradient.min()), dtype='uint8')
    #         print(gradient.shape)
    #         # 使用pyplot画图
    #         plt.subplot(121)
    #         plt.imshow(image)
    #         plt.subplot(122)
    #         plt.imshow(gradient, cmap=plt.cm.gray)
    #         plt.savefig('results/epoch%d/result_%d.png' % (epoch, batch))
        
    # def observe_hidden_distribution(self, batch_size=128, image_h=32, image_w=32, n_channel=3,
    #                                 num_test=10, epoch=1):
    #     if not os.path.exists('results/epoch%d/' % (epoch)):
    #         os.makedirs('results/epoch%d/' % (epoch))
    #     saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    #     sess = tf.Session()
    #     # 读取模型
    #     model_path = 'backup/cifar10/model_%d.ckpt' % (epoch)
    #     if os.path.exists(model_path+'.index'):
    #         saver.restore(sess, model_path)
    #         print('read model from %s' % (model_path))
    #     else:
    #         sess.run(tf.global_variables_initializer())
    #     # 获取图像并计算梯度
    #     for batch in range(num_test):
    #         batch_image, batch_label = cifar10.test.next_batch(batch_size)
    #         result = sess.run([self.nobn_conv1, self.bn_conv1, self.nobn_conv2, self.bn_conv2,
    #                            self.nobn_conv3, self.bn_conv3, self.nobn_fc1, self.nobn_fc1,
    #                            self.nobn_softmax, self.bn_softmax],
    #                           feed_dict={self.images:batch_image, self.labels:batch_label,
    #                                      self.keep_prob:0.5})
    #         distribution1 = result[0][:,0].flatten()
    #         distribution2 = result[1][:,0].flatten()
    #         distribution3 = result[2][:,0].flatten()
    #         distribution4 = result[3][:,0].flatten()
    #         distribution5 = result[4][:,0].flatten()
    #         distribution6 = result[5][:,0].flatten()
    #         distribution7 = result[6][:,0].flatten()
    #         distribution8 = result[7][:,0].flatten()
    #         plt.subplot(241)
    #         plt.hist(distribution1, bins=50, color='#1E90FF')
    #         plt.title('convolutional layer 1')
    #         plt.subplot(242)
    #         plt.hist(distribution3, bins=50, color='#1C86EE')
    #         plt.title('convolutional layer 2')
    #         plt.subplot(243)
    #         plt.hist(distribution5, bins=50, color='#1874CD')
    #         plt.title('convolutional layer 3')
    #         plt.subplot(244)
    #         plt.hist(distribution7, bins=50, color='#5CACEE')
    #         plt.title('full connection layer')
    #         plt.subplot(245)
    #         plt.hist(distribution2, bins=50, color='#00CED1')
    #         plt.title('batch normalized')
    #         plt.subplot(246)
    #         plt.hist(distribution4, bins=50, color='#48D1CC')
    #         plt.title('batch normalized')
    #         plt.subplot(247)
    #         plt.hist(distribution6, bins=50, color='#40E0D0')
    #         plt.title('batch normalized')
    #         plt.subplot(248)
    #         plt.hist(distribution8, bins=50, color='#00FFFF')
    #         plt.title('batch normalized')
    #         plt.show()




