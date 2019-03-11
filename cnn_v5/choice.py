#
# ##############################    用 mnist数据集 普通cnn模型 改这里    ################################
# import config.config_mnist as cfg
# from tensorflow.examples.tutorials.mnist import input_data as create_inputs
# from model.basic_cnn import cnn_mnist as model
# restore_model=False
# #############################      end    #######################################
#
# ##############################    用 cifar10数据集 普通cnn模型改这里    ################################
# import config.config_cifar10 as cfg
# from tools.use_cifar10 import prepare_train_data as create_inputs
# from model.basic_cnn import cnn_cifar10 as model
# restore_model=False
# #############################      end    #######################################
#
# ###############################    用 asl数据集 普通cnn模型 改这里    ################################
# from tools.use_asl_tfrecord import create_inputs_asl as create_inputs
# import config.config_asl as cfg
# from model.basic_cnn import cnn_L4 as model
# restore_model=False
# ##############################      end    #######################################

##############################    用 isl数据集 普通cnn模型 改这里    ################################
# from tools.use_isl_tfrecord import create_inputs_isl as create_inputs
# import config.config_isl as cfg
# from model.basic_cnn import cnn_L4 as model
# restore_model=False
#############################      end    #######################################


# ###############################    用 asl数据集 densenet模型改这里    ################################
from tools.use_asl_tfrecord import create_inputs_asl as create_inputs
import config.config_dense_asl as cfg
from model.densenet import dense_net as model
restore_model=False
# ##############################      end    #######################################




#让上面的代码亮起来
create_inputs = create_inputs
cfg = cfg
model =model
restore_model=restore_model