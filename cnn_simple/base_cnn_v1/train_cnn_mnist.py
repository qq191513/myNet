import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from basic_cnn import cnn
import config as cfg
import os
ckpt =cfg.ckpt
# 每个批次的大小
batch_size = cfg.batch_size
session_config = cfg.set_gpu()

with tf.Session(config = session_config) as sess:
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # 计算有多少批次
    n_batch = mnist.train.num_examples // batch_size
    # 构建网络
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    prediction = cnn(x, keep_prob=0.8)
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
    n_batch=50
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 训练一个step
            _, loss_value,acc_value, summary_str ,step= sess.run(
                [train_step, loss,accuracy, summary_op,global_step],
                feed_dict={x: batch_xs, y: batch_ys})
            # 显示结果batch_size
            cfg.print_message(step,loss_value,acc_value)
            # 保存summary
            if (step + 1) % 20 == 0:
                summary_writer.add_summary(summary_str, step)

        # 保存model
        if (((epoch + 1) % 5)) == 0:
            print('saving movdel.......')
            saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch)), global_step=global_step)

    cfg.stop_threads(coord,threads)