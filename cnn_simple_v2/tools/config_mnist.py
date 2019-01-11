import tensorflow as tf
import os

###############################    mnist 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
project_name = 'cnn_simple_v2'
dataset_name = 'mnist'
model_name = 'basic_cnn'
batch_size= 32
epoch  = 150
input_shape= (batch_size,28,28,1)
num_class = 10
dataset_root =os.path.join(project_root,'data','MNIST_data')
##############################      end    ########################################

logdir = os.path.join(project_root,'output',project_name,model_name,dataset_name,'logdir')
ckpt =os.path.join(project_root,'output',project_name,model_name,dataset_name)

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


def restore_model(sess,ckpt,restore_model):
    # 3、恢复model，放在上面的全局初始化和启动数据线程函数之后
    """Set Saver."""
    var_to_save = [v for v in tf.global_variables(
    ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
    saver = tf.train.Saver(var_list=var_to_save, max_to_keep=5)
    if restore_model:
        model_file = tf.train.latest_checkpoint(ckpt)
        saver.restore(sess, model_file)
    return saver

def stop_threads(coord,threads):
    # 4、程序终止 （该句要放到with graph和with sess 区域之内才行）
    coord.request_stop()
    coord.join(threads)

def set_optimizer(num_batches_per_epoch = None,loss = None):
    # 1、定义global_step
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # 2、定义优化器
    # （1）使用腐蚀因子
    lrn_rate = tf.maximum(tf.train.exponential_decay(
        1e-4, global_step, num_batches_per_epoch, 0.96), 1e-6)
    tf.summary.scalar('learning_rate', lrn_rate)
    opt = tf.train.AdamOptimizer(learning_rate =lrn_rate)  # lrn_rate
    # （2）手工
    # opt = tf.train.AdamOptimizer(0.0001).minimize(loss)
    # opt = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    # train_op = opt

    # 3、计算梯度
    grad = opt.compute_gradients(loss)

    # 4、检查梯度是否正常
    # See: https://stackoverflow.com/questions/40701712/how-to-check-nan-in-gradients-in-tensorflow-when-updating
    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                  for g, _ in grad if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]

    # 4.1、如果梯度正常
    with tf.control_dependencies(grad_check):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # 4.2、先执行update_ops
        with tf.control_dependencies(update_ops):
            # 4.3、再进行反向传播更新权值
            train_step = opt.apply_gradients(grad, global_step=global_step)

    return global_step,train_step

def set_summary(sess,summary_dict):
    for key,value in summary_dict.items():
        tf.summary.scalar(key, value)

    summary_op = tf.summary.merge_all()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir , graph=sess.graph)
    return summary_writer,summary_op



def cross_entropy_loss(logits,label):
    # 唯一的区别是非sparse的labels是one - hot类型。label是one-hot类型且是浮点数，如tf.constant([[0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]])
    # sparse的labels是int类型，
    # 注！logits如tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])，
    # logits不可经过tf.argmax之类的稠密化，
    # 也不能传进来之前已经tf.nn.softmax(logits)到了下面又来一次softmax，这样两次标准化会得到错误的los结果,虽然也有训练效果，但不知最后会造成怎样的影响

    ############### 方式1、手动算出代价函数
    # y = tf.nn.softmax(logits)
    # y_ =label
    # tf_log = tf.log(y)
    # pixel_wise_mult = tf.multiply(y_, tf_log)
    # cross_entropy_batch = -tf.reduce_sum(pixel_wise_mult)
    # cross_entropy = tf.reduce_mean(cross_entropy_batch)
    ############### end

    ############### 方式2、使用tf.nn.softmax_cross_entropy_with_logits算出代价函数
    # cross_entropy = tf.reduce_sum(
    #     tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    ############### end

    ############### 方式3、使用tf.nn.sparse_softmax_cross_entropy_with_logits
    dense_y = tf.arg_max(label, 1)     #如果label是onehot，则将标签稠密化
    cross_entropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=dense_y, logits=logits))
    ############### end

    ############### 方式4、二次代价函数
    # cross_entropy = tf.reduce_mean(tf.square(label - logits))
    ############### end

    return cross_entropy


def get_acc(prediction,y):
    # 结果存放在一个布尔类型列表中, tf.argmax返回一维张量中最大的值所在的位置，
    # 就是返回识别出来最可能的结果
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 求准确率，tf.case()把bool转化为float
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def print_message(step,loss_value,acc_value):
    message = ('step=%d ' % step + ' loss=%0.3f' % loss_value + ' acc=%0.3f' % acc_value)
    print(message)