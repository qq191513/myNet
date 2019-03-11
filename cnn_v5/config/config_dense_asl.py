import tensorflow as tf
import os

###############################     cifar10 改这里    ####################################
project_root = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/'
dataset_path = '/home/mo/work/data_set/asl/'
output_path = '/home/mo/work/output'
branch_name = 'cnn_v5'
model_name = 'densenet'
dataset_name = 'asl'
batch_size= 32
epoch  = 150
save_epoch_n = 10
num_class = 36
input_shape= [batch_size,28,28,3]
label_shape= [batch_size,num_class]
train_number = 2165
test_number = 350
lr_range=(1e-3,1e-6,0.96)
##############################      end    ########################################

ckpt =os.path.join(output_path,branch_name,model_name + '_' + dataset_name)
logdir = os.path.join(ckpt,'logdir')

