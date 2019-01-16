import tensorflow as tf
import sys
sys.path.append('../')
from tools.use_asl_tfrecord import create_inputs_asl
import tools.config_asl as cfg
import  numpy as np
import tools.development_kit as dk
###############################     改这里    ####################################
from model.basic_cnn import cnn_asl_L4
ckpt =cfg.ckpt
dataset_path =cfg.dataset_path
batch_size = cfg.batch_size
input_shape = cfg.input_shape
class_num = cfg.num_class
train_number = cfg.train_number
test_number = cfg.test_number
is_train = False
restore_model  = True
logdir = cfg.logdir
epoch = 20
##############################      end    ########################################
n_batch_train = int(train_number //batch_size)
n_batch_test = int(test_number //batch_size)
session_config = dk.set_gpu()

with tf.Session(config = session_config) as sess:
    test_x, test_y = create_inputs_asl(is_train)
    test_y = tf.one_hot(test_y, depth=class_num, axis=1, dtype=tf.float32)
    # 构建网络
    x = tf.placeholder(tf.float32, shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])
    y = tf.placeholder(tf.float32, shape=[input_shape[0], class_num])
    prediction = cnn_asl_L4(x, keep_prob=1)
    # 求acc
    accuracy = dk.get_acc(prediction, y)
    # 初始化变量
    coord, threads = dk.init_variables_and_start_thread(sess)
    # 恢复model
    saver = dk.restore_model(sess,ckpt,restore_model =restore_model)
    start_epoch = 0
    acc_list = []
    print('train_number:',train_number)
    print('test_number:',test_number)
    if is_train:
        n_batch_total = n_batch_train
    else:
        n_batch_total = n_batch_test
    for epoch_n in range(start_epoch,epoch):
        for n_batch in range(n_batch_total):
            batch_x, batch_y = sess.run([test_x, test_y])
            # 训练一个step
            acc_value= sess.run(accuracy,feed_dict={x: batch_x, y: batch_y})
            # 显示结果batch_size
            print('epoch_n:',epoch_n,' n_batch:',n_batch,' acc_value:',acc_value)
            acc_list.append(acc_value)
    result = np.mean(acc_list)
    print('final result: ',result)
    dk.stop_threads(coord,threads)