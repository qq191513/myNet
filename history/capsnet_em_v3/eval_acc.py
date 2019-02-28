import tensorflow as tf
from config import cfg, get_coord_add, get_dataset_size_train, get_dataset_size_test, get_num_classes, get_create_inputs
import time
import os

import tensorflow.contrib.slim as slim
from visual_tool import print_tensor
import logging
import daiquiri

daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)
####################   改这里  ##########################################
import capsnet_em as net
ckpt = 'logdir/caps/asl/'
dataset_name = 'asl'
####################   end       ########################################
def main(args):
    tf.set_random_seed(1234)
    coord_add = get_coord_add(dataset_name)
    dataset_size_train = get_dataset_size_train(dataset_name)
    dataset_size_test = get_dataset_size_test(dataset_name)
    num_classes = get_num_classes(dataset_name)
    create_inputs = get_create_inputs(
        dataset_name, is_train=False, epochs=cfg.epoch)

    with tf.Graph().as_default():
        num_batches_test = int(dataset_size_test / cfg.batch_size * 0.5)
        batch_x, batch_labels = create_inputs()
        output, pose_out = net.build_arch(batch_x, coord_add,is_train=False, num_classes=num_classes)
        tf.logging.debug(pose_out.get_shape())

        batch_acc = net.test_accuracy(output, batch_labels)
        saver = tf.train.Saver()
        session_config = tf.ConfigProto(
            device_count={'GPU': 0},
            gpu_options={'allow_growth': 1,
                         # 'per_process_gpu_memory_fraction': 0.1,
                         'visible_device_list': '0'},
            allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            mode_file = tf.train.latest_checkpoint(ckpt)
            saver.restore(sess, mode_file)

            accuracy_sum = 0
            for i in range(num_batches_test):
                batch_acc_v = sess.run(
                    [batch_acc])
                accuracy_sum += batch_acc_v[0]
                print(accuracy_sum)


            ave_acc = accuracy_sum / num_batches_test
            print('the average accuracy is %f' % ave_acc)


if __name__ == "__main__":
    tf.app.run()
