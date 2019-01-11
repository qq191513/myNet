import tensorflow as tf
import sys
sys.path.append('../')
from tools.cifar10 import *
from model.basic_cnn import cnn_cifar10
import tools.config_cifar10 as cfg
import os

###############################     改这里    ####################################
ckpt =cfg.ckpt
dataset_path =cfg.dataset_path
# 每个批次的大小
batch_size = cfg.batch_size
input_shape = cfg.input_shape
class_num = cfg.num_class
epoch = cfg.epoch
##############################      end    ########################################

os.makedirs(ckpt,exist_ok=True)
session_config = cfg.set_gpu()
with tf.Session(config = session_config) as sess:
    train_x, train_y, test_x, test_y = prepare_data(dataset_path)
    train_x, test_x = color_preprocessing(train_x, test_x)    # 计算有多少批次
    n_batch = len(train_x) // batch_size

    # 构建网络
    x = tf.placeholder(tf.float32, shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])
    y = tf.placeholder(tf.float32, shape=[input_shape[0], class_num])
    prediction = cnn_cifar10(x, keep_prob=0.8)
    # 求loss
    loss = cfg.cross_entropy_loss(prediction, y)
    # 设置优化器
    global_step, train_step = cfg.set_optimizer(num_batches_per_epoch=n_batch, loss=loss)
    # 求acc
    accuracy = cfg.get_acc(prediction, y)
    # 初始化变量
    coord, threads = cfg.init_variables_and_start_thread(sess)
    # 恢复model
    saver = cfg.restore_model(sess,ckpt,restore_model =False)
    # 设置训练日志
    summary_dict = {'loss':loss,'accuracy':accuracy}
    summary_writer, summary_op = cfg.set_summary(sess,summary_dict)
    # epoch=50
    for epoch_n in range(epoch):
        pre_index = 0
        for batch in range(n_batch):
            if pre_index+batch_size < 50000 :
                batch_x = train_x[pre_index : pre_index+batch_size]
                batch_y = train_y[pre_index : pre_index+batch_size]
            else :
                batch_x = train_x[pre_index : ]
                batch_y = train_y[pre_index : ]
            # 训练一个step
            _, loss_value,acc_value, summary_str ,step= sess.run(
                [train_step, loss,accuracy, summary_op,global_step],
                feed_dict={x: batch_x, y: batch_y})
            # 显示结果batch_size
            cfg.print_message(step,loss_value,acc_value)
            # 保存summary
            if (step + 1) % 20 == 0:
                summary_writer.add_summary(summary_str, step)

            pre_index += batch_size

        # 保存model
        if (((epoch_n + 1) % 5)) == 0:
            print('saving movdel.......')
            saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

    cfg.stop_threads(coord,threads)