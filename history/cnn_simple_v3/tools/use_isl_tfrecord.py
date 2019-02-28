import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
import os
import config.config_isl as cfg

preprocess_paraments={}
example_name = {}

##########################要改的东西#######################################
#tfrecords文件的路径
tfrecord_path = '/home/mo/work/data_set/italy_tf/'

#根据关键字搜索tfrecord_path目录下的所有相关文件
train_keywords = 'train'
test_keywords = 'validation'
labels_txt_keywords = 'labels.txt'

# 解码部分：填入解码键值和原图大小以便恢复
example_name['image'] = 'image/encoded'  #主要是这个(原图)p
example_name['label'] = 'image/class/label' #主要是这个(标签)
origenal_size =[32,32,1] #要还原原先图片尺寸

#预处理方式
to_random_brightness = True
to_random_contrast = True
to_resize_images = False
resize_size =[20,20]
to_random_crop = True
crop_size= [28, 28, 1]

#多队列、多线程、batch读图部分
num_threads = 8
batch_size = cfg.batch_size
shuffle_batch =True
#训练多少轮，string_input_producer的num_epochs就写多少，
# 否则会爆出OutOfRangeError的错误（意思是消费量高于产出量）
num_epochs = cfg.epoch

#显示方式
cv2_show = False  # 用opencv显示或plt显示
#######################  end  ############################################

def ReadTFRecord(tfrecords,example_name):
    if len(tfrecords) == 1:
        record_queue = tf.train.string_input_producer(tfrecords,num_epochs=num_epochs+1)#只有一个文件，谈不上打乱顺序
    else:
        # shuffle=False，num_epochs为3，即每个文件复制成3份，再打乱顺序，否则按原顺序
        record_queue = tf.train.string_input_producer(tfrecords,shuffle=True, num_epochs=num_epochs+1)

    reader = tf.TFRecordReader()
    key, value = reader.read(record_queue)
    features = tf.parse_single_example(value,
            features={
                # 取出key为img_raw和label的数据,尤其是int位数一定不能错!!!
                example_name['image']: tf.FixedLenFeature([],tf.string),
                example_name['label']: tf.FixedLenFeature([], tf.int64)
            })

    img = tf.decode_raw(features[example_name['image']], tf.uint8)
    # 注意定义的为int多少位就转换成多少位,否则容易出错!!

    if len(origenal_size) == 2:
        w, h = origenal_size[0],origenal_size[1]
    else:
        w, h, c = origenal_size[0],origenal_size[1],origenal_size[2]
    img = tf.reshape(img, [w, h, c])

    # 不清楚为何加了这个tf.cast会让cv2.imshow显示不正常，图片变成黑白二值图，模糊不清
    img = tf.cast(img, tf.float32)

    label = tf.cast(features[example_name['label']], tf.int64)
    label = tf.cast(label, tf.int32)

    return img, label

def preprocess_data(is_train,image, label):
    if is_train:

        if to_random_brightness:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        if to_random_contrast:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        if to_resize_images:
            # 只有method = 1没有被破坏最严重
            image = tf.image.resize_images(image, resize_size,method=1)
        if to_random_crop:
            image = tf.random_crop(image, crop_size)


    else:
        if to_resize_images:
            image = tf.image.resize_images(image, [28, 28])
        if to_random_crop:
            image = tf.random_crop(image, crop_size)

    return image, label

def feed_data_method(image,label):
    if shuffle_batch:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=batch_size*64,
            min_after_dequeue=batch_size*32,
            allow_smaller_final_batch=False)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=batch_size*64,
            allow_smaller_final_batch=False)
    return images, labels

def plt_imshow_data(data):
    #调成标准格式和标准维度，免得爆BUG
    data = np.asarray(data)

    if data.ndim == 3:
        if data.shape[2] == 1:
            data = data[:, :, 0]
    plt.imshow(data)
    plt.show()
    time.sleep(2)

def get_files_list(path):
    # work：获取所有文件的完整路径
    files_list = []
    for parent,dirnames,filenames in os.walk(path):
        for filename in filenames:
            files_list.append(os.path.join(parent,filename))
    return files_list

#根据关键字筛选父目录下需求的文件，按列表返回全部完整路径
def search_keyword_files(path,keyword):
    keyword_files_list = []
    files_list = get_files_list(path)
    for file in files_list:
        if keyword in file:
            keyword_files_list.append(file)
    return keyword_files_list

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

def show_loaded(data_tfrecord=None):
    print('load tfrecord:')
    for each in data_tfrecord:
        print(each)

def create_inputs_isl(is_train):
    if is_train:
        data_tfrecord = search_keyword_files(tfrecord_path,train_keywords)
    else:
        data_tfrecord = search_keyword_files(tfrecord_path,test_keywords)
    show_loaded(data_tfrecord)
    image, label = ReadTFRecord(data_tfrecord,example_name)          #恢复原始数据
    image, label = preprocess_data(is_train,image, label)            #预处理方式
    images,labels = feed_data_method(image, label)                   #喂图方式

    return images,labels

if  __name__== '__main__':
    images, labels = create_inputs_isl(is_train = True)
    labels_txt = search_keyword_files(tfrecord_path, labels_txt_keywords)
    labels_maps = read_label_txt_to_dict(labels_txt=labels_txt[0])   #标签映射
    #观察自己设置的参数是否符合心意，合适的话在别的项目中直接调用 create_inputs_xxx() 函数即可喂数据
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输出100个batch观察
        batch_size_n = 5  #观察batch_size的第n张图片
        if cv2_show:
            for i in range(100):
                x, y = sess.run([images, labels])
                label = y[batch_size_n]
                label = str(label)
                label_name = labels_maps[label]
                title = 'label:{}'.format(label_name)
                print('image_shape:',x.shape, 'label_shape:', y.shape,title)
                cv2.namedWindow(title, 0)
                cv2.startWindowThread()
                cv2.imshow(title, x[batch_size_n])
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
        else:
            for i in range(100):
                x, y = sess.run([images, labels])
                label = y[batch_size_n]
                label = str(label)
                label_name =labels_maps[label]
                title = 'label:{}'.format(label_name)
                print('image_shape:',x.shape, 'label_shape:', y.shape,title)

                plt_imshow_data(x[batch_size_n])


        coord.request_stop()
        coord.join(threads)
