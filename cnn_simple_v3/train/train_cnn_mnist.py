import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')
import tools.config_mnist as cfg
import tools.development_kit as dk
import os
###############################     改这里    ####################################
from model.basic_cnn import cnn_mnist
ckpt =cfg.ckpt
# 每个批次的大小
batch_size = cfg.batch_size
input_shape = cfg.input_shape
class_num = cfg.num_class
dataset_path =cfg.dataset_path
epoch = cfg.epoch
logdir = cfg.logdir
##############################      end    ########################################


#技术暂时不够、#报错就手工下载，经常难以通过代码下载，命令下载总是各种BUG,先手工wget命令下载到MNIST_data文件夹
# downlaod_list = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
#             ]

os.makedirs(ckpt,exist_ok=True)
session_config = dk.set_gpu()
with tf.Session(config = session_config) as sess:
    mnist = input_data.read_data_sets(dataset_path, one_hot=True)
    # 计算有多少批次
    n_batch = mnist.train.num_examples // batch_size
    # 构建网络
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, class_num])
    prediction = cnn_mnist(x, keep_prob=0.8)
    # 求loss
    loss = dk.cross_entropy_loss(prediction, y)
    # 设置优化器
    global_step, train_step = dk.set_optimizer(num_batches_per_epoch=n_batch, loss=loss)
    # 求acc
    accuracy = dk.get_acc(prediction, y)
    # 初始化变量
    coord, threads = dk.init_variables_and_start_thread(sess)
    # 恢复model
    saver = dk.restore_model(sess,ckpt,restore_model =False)
    # 设置训练日志
    summary_dict = {'loss':loss,'accuracy':accuracy}
    summary_writer, summary_op = dk.set_summary(sess,logdir,summary_dict)
    # epoch=50
    for epoch_n in range(epoch):
        pre_index = 0
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 训练一个step
            _, loss_value,acc_value, summary_str ,step= sess.run(
                [train_step, loss,accuracy, summary_op,global_step],
                feed_dict={x: batch_xs, y: batch_ys})
            # 显示结果batch_size
            dk.print_message(epoch_n,step,loss_value,acc_value)
            # 保存summary
            if (step + 1) % 20 == 0:
                summary_writer.add_summary(summary_str, step)

        # 保存model
        if (((epoch_n + 1) % 5)) == 0:
            print('saving movdel.......')
            saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

    dk.stop_threads(coord,threads)