import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#技术暂时不够、命令下载总是各种BUG,先手工wget命令下载到MNIST_data文件夹
# downlaod_list = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
#             'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
#             ]

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder，x是图片样本，y是输出的结果
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔类型列表中, tf.argmax返回一维张量中最大的值所在的位置，就是返回识别出来最可能的结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率，tf.case()把bool转化为float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy: " + str(acc))