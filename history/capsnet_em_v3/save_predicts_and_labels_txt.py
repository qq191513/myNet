import tensorflow as tf
import config as cfg
from sklearn.metrics import confusion_matrix
#用法介绍
# labels=["ant", "bird", "cat"]
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# C=confusion_matrix(y_true, y_pred,labels=labels)
# print(C)

import logging
import daiquiri
daiquiri.setup(level=logging.DEBUG)
logger = daiquiri.getLogger(__name__)
import capsnet_em as net
from confusion_matrix_API import plot_confusion
import matplotlib.pyplot as plt

####################   改这里  ##########################################
import capsnet_em as net
ckpt = 'logdir/caps/asl/'
dataset_name = 'asl'
#想要测试多少个batch
test_dataset_size = cfg.get_dataset_size_test('asl')
batch_size =cfg.batch_size
num_batches_test = 10 * (test_dataset_size // batch_size)
# num_batches_test = 10  #数量太少会报错，因为画图有的行是NaN值
recognize_data_dir ='../data/asl_tf'
recognize_labels_txt_keywords = 'labels.txt'
####################   end       ########################################

def conver_number_to_label_name(the_list,labels_maps):
    for idx,each in enumerate(the_list):
        the_list[idx] = labels_maps[str(each)]
    return the_list

def main(args):
    # 1、设置GPU模式
    session_config = cfg.set_gpu()

    with tf.Graph().as_default():

        # 2、设置随机种子、读取数据batch、类别数
        tf.set_random_seed(1234)
        coord_add = cfg.get_coord_add(dataset_name)
        num_classes = cfg.get_num_classes(dataset_name)
        labels_txt = cfg.search_keyword_files(recognize_data_dir, recognize_labels_txt_keywords)
        labels_maps = cfg.read_label_txt_to_dict(labels_txt[0])


        with tf.Session(config=session_config) as sess:

            create_inputs = cfg.get_create_inputs(dataset_name, is_train=False, epochs=cfg.epoch)
            batch_x, batch_labels = create_inputs()


            # 3、初始化网络
            output, pose_out = net.build_arch(batch_x, coord_add, is_train=False, num_classes=num_classes)
            tf.logging.debug(pose_out.get_shape())
            results, labels = net.batch_results_and_labels(output, batch_labels)

            # 4、全局初始化和启动数据线程 （要放在初始化网络之后）
            coord, threads = cfg.init_variables_and_start_thread(sess)

            # 5、恢复model
            cfg.restore_model(sess, ckpt)

            # 6、求出全部预测值和标签list
            np_predicts_list = []
            np_lables_list = []
            for i in range(num_batches_test):
                np_results,np_labels = sess.run(
                    [results, labels])
                print(np_results)
                print(np_labels)
                np_predicts_list.extend(np_results)
                np_lables_list.extend(np_labels)

            np_predicts_list_str = str(np_predicts_list)
            np_lables_list_str = str(np_lables_list)
            with open('predicts_and_labels.txt','w') as f:
                f.write('predicts\r\n')
                f.write(np_predicts_list_str + '\r\n')
                f.write('labels\r\n')
                f.write(np_lables_list_str + '\r\n')

            cfg.stop_threads(coord,threads)


if __name__ == "__main__":
    tf.app.run()










