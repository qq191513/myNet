import tensorflow as tf
import os

###############################     cifar10 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
dataset_path = '/home/mo/work/data_set/isl/'
output_path = '/home/mo/work/output'
branch_name = 'cnn_simple_v3'
model_name = 'cnn_L4'
dataset_name = 'isl'
batch_size= 32
epoch  = 150
input_shape= (batch_size,28,28,1)
num_class = 22
train_number = 8080
test_number = 900
##############################      end    ########################################

ckpt =os.path.join(output_path,branch_name,model_name + '_' + dataset_name)
logdir = os.path.join(ckpt,'logdir')

