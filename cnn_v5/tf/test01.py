import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#定义神经层
def add_layer(inputs,in_size,out_size,activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

#定义训练的数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#定义隐藏层、输出层
#给train_step输入值,在声明完占位符之后用tf.to_double（xs）
xs = tf.placeholder(tf.float32,x_data.shape)
ys = tf.placeholder(tf.float32,y_data.shape)
tf.to_double(xs)
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

#预测Loss并用梯度下降法进行学习
loss =tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化所有变量
init = tf.initialize_all_variables()
#开始运算
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


for i in range(1000):
    #训练
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        #观察每一步的优化

        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data,prediction_value,'r',lw=5)
        plt.pause(0.1)
