import tensorflow as tf
import numpy as np
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.framework import arg_scope
from collections import OrderedDict

# Hyperparameter
growth_k = 24
filters = growth_k
nb_block = 2 # how many (dense block + Transition Layer) ?
dropout_rate = 0.2
training = tf.cast(True, tf.bool)
end_point = OrderedDict()

# Momentum Optimizer will use
# init_learning_rate = 1e-4
# epsilon = 1e-4 # AdamOptimizer epsilon
# nesterov_momentum = 0.9
# weight_decay = 1e-4


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x,class_num) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



def transition_layer(x, scope):
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)
        x = Average_pooling(x, pool_size=[2,2], stride=2)

        return x

def dense_block(input_x, nb_layers, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

        layers_concat.append(x)

        for i in range(nb_layers - 1):
            x = Concatenation(layers_concat)
            x = bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
            layers_concat.append(x)

        x = Concatenation(layers_concat)
        return x

def bottleneck_layer(x, scope):
    # print(x)
    with tf.name_scope(scope):
        x = Batch_Normalization(x, training=training, scope=scope+'_batch1')
        x = Relu(x)
        x = conv_layer(x, filter=4 * filters, kernel=[1,1], layer_name=scope+'_conv1')
        x = Drop_out(x, rate=dropout_rate, training=training)

        x = Batch_Normalization(x, training=training, scope=scope+'_batch2')
        x = Relu(x)
        x = conv_layer(x, filter=filters, kernel=[3,3], layer_name=scope+'_conv2')
        x = Drop_out(x, rate=dropout_rate, training=training)

        # print(x)

        return x

def dense_net(input, input_shape,class_num):

    filters = growth_k
    end_point['input'] = input
    end_point['conv_layer'] = x = conv_layer(input, filter=2 * filters, kernel=[7, 7], stride=2, layer_name='conv0')
    # x = Max_Pooling(x, pool_size=[3,3], stride=2)

    """
    nb_blocks = nb_block
    for i in range(self.nb_blocks) :
        # 6 -> 12 -> 48
        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
        x = self.transition_layer(x, scope='trans_'+str(i))
    """

    end_point['dense_block_1'] = x = dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
    end_point['transition_layer_1'] = x = transition_layer(x, scope='trans_1')

    end_point['dense_block_2'] = x = dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
    end_point['transition_layer_2'] = x = transition_layer(x, scope='trans_2')

    end_point['dense_block_3'] = x = dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
    end_point['transition_layer_3'] = x = transition_layer(x, scope='trans_3')

    end_point['dense_block_final'] = x = dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

    # 100 Layer
    end_point['Batch_Normalization'] = x = Batch_Normalization(x, training=training, scope='linear_batch')
    end_point['Relu'] = x = Relu(x)
    end_point['Global_Average_Pooling'] = x = Global_Average_Pooling(x)
    end_point['flatten'] = x = flatten(x)
    end_point['Linear'] = y_predict = Linear(x,class_num)



    return y_predict,end_point