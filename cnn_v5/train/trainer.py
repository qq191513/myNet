import tensorflow as tf
import sys
sys.path.append('../')
from choice import create_inputs
from choice import cfg
import os
import tools.development_kit as dk
import time
from choice import model
from choice import restore_model
###############################    改这里    ####################################

ckpt =cfg.ckpt
batch_size = cfg.batch_size
input_shape = cfg.input_shape
label_shape = cfg.label_shape
class_num = cfg.num_class
epoch = cfg.epoch
train_number = cfg.train_number
test_number = cfg.test_number
is_train = True
save_epoch_n = cfg.save_epoch_n  #每多少epoch保存一次
logdir = cfg.logdir
lr_range =cfg.lr_range
##############################      end    ########################################
n_batch_train = int(train_number //batch_size)
n_batch_test = int(test_number //batch_size)
os.makedirs(ckpt,exist_ok=True)
session_config = dk.set_gpu()
def train_model():
    with tf.Session(config = session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=label_shape)
        # 构建网络和预测
        prediction ,endpoint = model(x, input_shape,class_num)
        # 打印模型结构
        dk.print_model_struct(endpoint)
        # 求loss
        loss = dk.cross_entropy_loss(prediction, y)
        # 设置优化器
        global_step, train_step = dk.set_optimizer(lr_range,num_batches_per_epoch=n_batch_train, loss=loss)
        # 求acc
        accuracy = dk.get_acc(prediction, y)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 设置训练日志
        summary_dict = {'loss':loss,'accuracy':accuracy}
        summary_writer, summary_op = dk.set_summary(sess,logdir,summary_dict)
        # 恢复model
        saver,start_epoch = dk.restore_model(sess,ckpt,restore_model =restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 训练loop
        total_step = n_batch_train * epoch
        for epoch_n in range(start_epoch,epoch):
            acc_value_list = []
            since = time.time()
            for n_batch in range(n_batch_train):
                batch_x, batch_y = sess.run([train_x, train_y])
                # 训练一个step
                _, loss_value,acc_value, summary_str ,step= sess.run(
                    [train_step, loss,accuracy, summary_op,global_step],
                    feed_dict={x: batch_x, y: batch_y})
                # 显示结果batch_size
                dk.print_message(epoch_n,n_batch,n_batch_train,step,loss_value,acc_value)
                # 保存summary
                if (step + 1) % 20 == 0:
                    summary_writer.add_summary(summary_str, step)
                # 保存结果
                acc_value_list.append(acc_value)

            # 显示进度、耗时、最小最大平均值
            seconds_mean = (time.time() - since) / n_batch_train
            dk.print_progress_and_time_massge(seconds_mean, step, total_step, acc_value_list)

            # 保存model
            if (((epoch_n + 1) % save_epoch_n)) == 0:
                print('epoch_n :{} saving movdel.......'.format(epoch_n))
                saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)
        dk.stop_threads(coord,threads)