import tensorflow as tf
import sys
sys.path.append('../')
import os
import tools.development_kit as dk
###############################    用asl  改这里    ################################
# from tools.use_asl_tfrecord import create_inputs_asl as create_inputs
# import tools.config_asl as cfg
##############################      end    #######################################

###############################    用isl  改这里    ################################
from tools.use_isl_tfrecord import create_inputs_isl as create_inputs
import tools.config_isl as cfg
##############################      end    #######################################

###############################     set    ####################################
from model.basic_cnn import cnn_L4
ckpt =cfg.ckpt
dataset_path =cfg.dataset_path
batch_size = cfg.batch_size
input_shape = cfg.input_shape
class_num = cfg.num_class
epoch = cfg.epoch
train_number = cfg.train_number
test_number = cfg.test_number
is_train = True
restore_model  = False
save_epoch_n = 10  #每多少epoch保存一次
logdir = cfg.logdir
##############################      end    ########################################

n_batch_train = int(train_number //batch_size)
n_batch_test = int(test_number //batch_size)
os.makedirs(ckpt,exist_ok=True)
session_config = dk.set_gpu()

with tf.Session(config = session_config) as sess:
    train_x, train_y = create_inputs(is_train)
    train_y = tf.one_hot(train_y, depth=class_num, axis=1, dtype=tf.float32)

    # 构建网络
    x = tf.placeholder(tf.float32, shape=[input_shape[0], input_shape[1], input_shape[2], input_shape[3]])
    y = tf.placeholder(tf.float32, shape=[input_shape[0], class_num])
    prediction = cnn_L4(x, input_shape,class_num,keep_prob=0.8)
    # 求loss
    loss = dk.cross_entropy_loss(prediction, y)
    # 设置优化器
    global_step, train_step = dk.set_optimizer(num_batches_per_epoch=n_batch_train, loss=loss)
    # 求acc
    accuracy = dk.get_acc(prediction, y)
    # 初始化变量
    coord, threads = dk.init_variables_and_start_thread(sess)
    # 设置训练日志
    summary_dict = {'loss':loss,'accuracy':accuracy}
    summary_writer, summary_op = dk.set_summary(sess,logdir,summary_dict)
    # 恢复model
    saver = dk.restore_model(sess,ckpt,restore_model =restore_model)
    # 显示参数量
    dk.show_parament_numbers()
    start_epoch = 0
    if restore_model:
        step = sess.run(global_step)
        start_epoch = int(step/n_batch_train/save_epoch_n)*save_epoch_n
    for epoch_n in range(start_epoch,epoch):
        for n_batch in range(n_batch_train):
            batch_x, batch_y = sess.run([train_x, train_y])
            # 训练一个step
            _, loss_value,acc_value, summary_str ,step= sess.run(
                [train_step, loss,accuracy, summary_op,global_step],
                feed_dict={x: batch_x, y: batch_y})
            # 显示结果batch_size
            dk.print_message(epoch_n,step,loss_value,acc_value)
            # 保存summary
            if (step + 1) % 20 == 0:
                summary_writer.add_summary(summary_str, step)

        # 保存model
        if (((epoch_n + 1) % save_epoch_n)) == 0:
            print('epoch_n :{} saving movdel.......'.format(epoch_n))
            saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

    dk.stop_threads(coord,threads)