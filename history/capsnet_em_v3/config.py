import tensorflow as tf

flags = tf.app.flags
import os
import sys
sys.path.append('../')
####################   改这里  ##########################################
flags.DEFINE_integer('A', 136, 'number of channels in output from ReLU Conv1')
flags.DEFINE_string('dataset', 'data/asl', 'the path for dataset')
flags.DEFINE_boolean('is_train', True, 'train or predict phase')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('test_logdir', 'test_logdir', 'test logs directory')
####################   end       ########################################


############################
#    hyper parameters      #
############################
flags.DEFINE_float('ac_lambda0', 0.01, '\lambda in the activation function a_c, iteration 0')
flags.DEFINE_float('ac_lambda_step', 0.01,
                   'It is described that \lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.')


flags.DEFINE_integer('iter_routing', 2, 'number of iterations')
flags.DEFINE_float('m_schedule', 0.2, 'the m will get to 0.9 at current epoch')
flags.DEFINE_float('epsilon', 1e-9, 'epsilon')
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_boolean('weight_reg', True, 'train with regularization of weights')
flags.DEFINE_string('norm', 'norm2', 'norm type')
################################
#    structure parameters      #
################################

flags.DEFINE_integer('B', 8, 'number of capsules in output from PrimaryCaps')
flags.DEFINE_integer('C', 16, 'number of channels in output from ConvCaps1')
flags.DEFINE_integer('D', 16, 'number of channels in output from ConvCaps2')

############################
#   environment setting    #
############################
flags.DEFINE_string('dataset_fashion_mnist', 'data/fashion_mnist', 'the path for dataset')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_integer('batch_size', 32, 'batch size')
# flags.DEFINE_integer('epoch', 150, 'epoch')
batch_size= 32
epoch  = 150
cfg = tf.app.flags.FLAGS
logdir = 'logdir'


def get_coord_add(dataset_name: str):
    import numpy as np
    # TODO: get coord add for cifar10/100 datasets (32x32x3)
    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.),
               'asl': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'italy': ([[[8., 8.], [12., 8.], [16., 8.]],
                        [[8., 12.], [12., 12.], [16., 12.]],
                        [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               }
    coord_add, scale = options[dataset_name]

    coord_add = np.array(coord_add, dtype=np.float32) / scale

    return coord_add


def get_dataset_size_train(dataset_name: str):
    options = {'mnist': 55000, 'smallNORB': 23400 * 2, 'asl': 2165,'italy': 8080,
               'fashion_mnist': 55000, 'cifar10': 50000, 'cifar100': 50000}
    return options[dataset_name]


def get_dataset_size_test(dataset_name: str):
    options = {'mnist': 10000, 'smallNORB': 23400 * 2,'asl': 350,'italy': 900,
               'fashion_mnist': 10000, 'cifar10': 10000, 'cifar10': 10000}
    return options[dataset_name]


def get_num_classes(dataset_name: str):
    options = {'mnist': 10, 'smallNORB': 5, 'fashion_mnist': 10,
               'asl': 36,'italy': 22,'cifar10': 10, 'cifar100': 100}
    return options[dataset_name]


from utils import create_inputs_mnist, create_inputs_cifar10, create_inputs_cifar100
# import sys
# sys.path.append('/home/mo/work/dataset')
from data.use_asl_tfrecord import create_inputs_asl
from data.use_italy_tfrecord import create_inputs_italy
def get_create_inputs(dataset_name: str, is_train: bool, epochs: int):
    options = {'mnist': lambda: create_inputs_mnist(is_train),
               'asl': lambda: create_inputs_asl(is_train),
               'italy': lambda: create_inputs_italy(is_train),
               'fashion_mnist': lambda: create_inputs_mnist(is_train),

               'cifar10': lambda: create_inputs_cifar10(is_train)}
    return options[dataset_name]

def get_files_list(path):
    # work：获取所有文件的完整路径
    files_list = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            files_list.append(os.path.join(parent,filename))
    return files_list

def read_label_txt_to_dict(labels_txt =None):
    if os.path.exists(labels_txt):
        labels_maps = {}
        with open(labels_txt) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line[:-1]  # 去掉换行符
                line_split = line.split(':')
                labels_maps[line_split[0]] = line_split[1]
        return labels_maps
    return None

#根据关键字筛选父目录下需求的文件，按列表返回全部完整路径
def search_keyword_files(path,keyword):
    keyword_files_list = []
    files_list = get_files_list(path)
    for file in files_list:
        if keyword in file:
            keyword_files_list.append(file)
    return keyword_files_list

def set_gpu():
    # 1、设置GPU模式
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)
    return  session_config

def init_variables_and_start_thread(sess):
    # 2、全局初始化和启动数据线程 （要放在初始化网络之后）
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return coord,threads


def restore_model(sess,ckpt):
    # 3、恢复model，放在上面的全局初始化和启动数据线程函数之后
    """Set Saver."""
    var_to_save = [v for v in tf.global_variables(
    ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
    saver = tf.train.Saver(var_list=var_to_save, max_to_keep=5)
    mode_file = tf.train.latest_checkpoint(ckpt)
    saver.restore(sess, mode_file)

def stop_threads(coord,threads):
    # 4、程序终止 （该句要放到with graph和with sess 区域之内才行）
    coord.request_stop()
    coord.join(threads)
def set_optimizer(num_batches_per_epoch = None):
    """Get global_step."""
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)


    """Use exponential decay leanring rate?"""
    lrn_rate = tf.maximum(tf.train.exponential_decay(
        1e-1, global_step, num_batches_per_epoch, 0.95), 1e-5)
    tf.summary.scalar('learning_rate', lrn_rate)
    opt = tf.train.AdamOptimizer()  # lrn_rate


