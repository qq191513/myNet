import tensorflow as tf
import os

###############################     cifar10 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
output_path = '/home/mo/work/output'
dataset_path = '/home/mo/work/data_set/cifar-10-batches-py/'
branch_name = 'cnn_simple_v3'
model_name = 'cnn_cifar10'
dataset_name = 'cifar10'
batch_size= 32
epoch  = 150
input_shape= (batch_size,32,32,3)
num_class = 10
##############################      end    ########################################

ckpt =os.path.join(output_path,branch_name,model_name,dataset_name)
logdir = os.path.join(ckpt,'logdir')
