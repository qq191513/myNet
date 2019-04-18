import tensorflow as tf
import os

###############################    mnist 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
output_path = '/home/mo/work/output'
branch_name = 'cnn_v5'
dataset_path = '/home/mo/work/data_set/MNIST_data/'
model_name = 'cnn_mnist'
dataset_name = 'mnist'
batch_size= 32
epoch  = 150
num_class = 10
lr_range=(1e-3,1e-6,0.96)
input_shape= (batch_size,28,28,1)
label_shape= [batch_size,num_class]
train_number = 50000
test_number = 10000
##############################      end    ########################################

ckpt =os.path.join(output_path,branch_name,model_name)
logdir = os.path.join(ckpt,'logdir')
