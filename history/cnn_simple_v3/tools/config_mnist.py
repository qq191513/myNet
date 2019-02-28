import tensorflow as tf
import os

###############################    mnist 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
output_path = '/home/mo/work/output'
branch_name = 'cnn_simple_v3'
dataset_path = '/home/mo/work/data_set/MNIST_data/'
model_name = 'cnn_mnist'
dataset_name = 'mnist'
batch_size= 32
epoch  = 150
input_shape= (batch_size,28,28,1)
num_class = 10
##############################      end    ########################################

ckpt =os.path.join(output_path,branch_name,model_name)
logdir = os.path.join(ckpt,'logdir')
