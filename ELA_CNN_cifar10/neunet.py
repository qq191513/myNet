# -*- coding: utf8 -*-
# author: ronniecao
import os
from src.data.cifar10 import Corpus
from keras import backend as K
K.clear_session()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import namedtuple
cifar10 = Corpus()

def my_cnn():
    from src.model.my_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v4/', batch_size=128, n_epoch=25)
    convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v4/', epoch=25, batch_size=128)

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v2/', batch_size=128, n_epoch=20)
    # convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v2/', epoch=5000, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def plain_cnn():
    from src.model.plain_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/cifar10-v16/', batch_size=128, n_epoch=500)
    # convnet.test(dataloader=cifar10,backup_path='backup/cifar10-v3/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def residual_net_L20():
    from src.model.residual_net_L20 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=20)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/residual_net_train_1/', batch_size=128, n_epoch=40)
    convnet.test(dataloader=cifar10,backup_path='backup/residual_net_train_1', epoch=40, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def my_residual_v1_N3L20():
    from src.model.my_residual_v1_N3L20 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=20)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_residual-v2/', batch_size=128, n_epoch=40)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual-v1_train_1/', epoch=40, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def my_residual_v2_N4L20():
    from src.model.my_residual_v2_N4L20 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=20)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_residual-v2_train_1/', batch_size=64, n_epoch=40)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual-v2_train_1/', epoch=40, batch_size=64)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def my_residual_v3_N5L20():
    from src.model.my_residual_v3_N5L20 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=20)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_residual-v3_train_1/', batch_size=64, n_epoch=40)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual-v3_train_1/', epoch=40, batch_size=64)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)






def residual_net_v4_N1L44():
    from src.model.my_residual_v4_N1L44 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/my_residual-v4_train_1/', batch_size=64, n_epoch=40)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual-v4_train_1/', epoch=40, batch_size=64)


def my_residual_v5_N3L44():
    from src.model.my_residual_v5_N3L44 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_residual-v5_train_1/', batch_size=64, n_epoch=30)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual-v5_train_1/', epoch=30, batch_size=64)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def my_residual_v6_RES1_CON14_CON8():
    from src.model.my_residual_v6_RES1_CON14_CON8 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_residual_v6_RES1_CON14_CON8/', batch_size=64, n_epoch=30)
    convnet.test(dataloader=cifar10,backup_path='backup/my_residual_v6_RES1_CON14_CON8/', epoch=30, batch_size=64)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)

def my_v7_N1_squeeze():
    from src.model.my_v7_N1_squeeze import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32)
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backup/my_v7_N1_squeeze/', batch_size=128, n_epoch=30)
    convnet.test(dataloader=cifar10, backup_path='backup/my_v7_N1_squeeze/', epoch=30, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)


def my_v8_N5_squeeze():
    from src.model.my_v8_N5_squeeze import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32)
    # convnet.debug()
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v8_N5_squeeze/', batch_size=128, n_epoch=30)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v8_N5_squeeze/', epoch=30, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)


def my_v9_N3_squeeze():
    from src.model.my_v9_N3_squeeze import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v9_N3_squeeze/', batch_size=128, n_epoch=1)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v9_N3_squeeze/', epoch=1, batch_size=128)

def my_v10_N3_plain_cnn_L14():
    from src.model.my_v10_N3_plain_cnn_L14 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v10_N3_plain_cnn_L14/', batch_size=64, n_epoch=200)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v10_N3_plain_cnn_L14/', epoch=200, batch_size=64)

def my_v11_N1_plain_cnn_L14():
    from src.model.my_v11_N1_plain_cnn_L14 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32, n_layers=44)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v11_N1_plain_cnn_L14/', batch_size=64, n_epoch=200)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v11_N1_plain_cnn_L14/', epoch=200, batch_size=64)

def my_v12_N3_reidual_L44():
    from src.model.my_v12_N3_reidual_L44 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v12_N3_reidual_L44/', batch_size=64, n_epoch=150)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v12_N3_reidual_L44/', epoch=150, batch_size=64)

def my_v13_N1_reidual_L44():
    from src.model.my_v13_N1_reidual_L44 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v13_N1_reidual_L44/', batch_size=64, n_epoch=150)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v13_N1_reidual_L44/', epoch=150, batch_size=64)

def my_v14_N5_reidual_L44():
    from src.model.my_v14_N5_reidual_L44 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=44)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v14_N5_reidual_L44/', batch_size=64, n_epoch=150)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v14_N5_reidual_L44/', epoch=150, batch_size=64)

def my_v15_N1_reidual_L32_google_v2():
    from src.model.my_v15_N1_R_Gv2 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v15_N1_reidual_L32/', batch_size=64, n_epoch=50)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v15_N1_reidual_L32/', epoch=50, batch_size=64)


def my_v16_N1_reidual_L32_google_v3():
    from src.model.my_v16_N1_R_Gv3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v16_N1_reidual_L32/', batch_size=64, n_epoch=50)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v16_N1_reidual_L32/', epoch=50, batch_size=64)

def my_v17_N1_reidual_L32_plain_L3():
    from src.model.my_v17_N1_R_L3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v17_N1_reidual_L32_plain_L3/', batch_size=64, n_epoch=50)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v17_N1_reidual_L32_plain_L3/', epoch=50, batch_size=64)

def my_v18_N3_Mix_before3():
    from src.model.my_v18_N3_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v18_N3_Mix_before3/', batch_size=64, n_epoch=50)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v18_N3_Mix_before3/', epoch=50, batch_size=64)

def my_v19_N9_out9_Mix_before3():
    from src.model.my_v19_N9_out4_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v19_N9_out9_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v19_N9_out9_Mix_before3/', epoch=50, batch_size=64)


def my_v20_N9_out5_Mix_before3():
    from src.model.my_v21_N9_out6_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v20_N9_out5_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v20_N9_out5_Mix_before3/', epoch=50, batch_size=64)

def my_v21_N9_out6_Mix_before3():
    from src.model.my_v21_N9_out6_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v21_N9_out6_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v20_N9_out5_Mix_before3/', epoch=50, batch_size=64)

def my_v22_N9_out7_Mix_before3():
    from src.model.my_v22_N9_out7_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v22_N9_out7_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v20_N9_out5_Mix_before3/', epoch=50, batch_size=64)

def my_v23_N9_out9_Mix_before3():
    from src.model.my_v23_N9_out9_Mix_before3 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v23_N9_out9_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v23_N9_out9_Mix_before3/', epoch=50, batch_size=64)



def my_v24_N9_out9_Mix_before3_S():
    from src.model.my_v24_N9_out9_Mix_before3_S import ConvNet
    debug_mode = True
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,debug_mode=debug_mode)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v24_N9_out9_Mix_before3_S/', batch_size=64, n_epoch=55)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =False,debug_mode =debug_mode)
    # convnet.test(dataloader=cifar10,backup_path='backup/my_v24_N9_out9_Mix_before3_S/', epoch=50, batch_size=64)

def my_v25_N9_out9_Mix_before3_ST2():
    from src.model.my_v25_N9_out9_Mix_before3_ST2 import ConvNet
    debug_mode = True
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,debug_mode=debug_mode)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v25_N9_out9_Mix_before3_ST2/', batch_size=64, n_epoch=55)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False, debug_mode=debug_mode)
    # convnet.test(dataloader=cifar10, backup_path='backup/my_v25_N9_out9_Mix_before3_ST2/', epoch=50, batch_size=64)
    #
def my_v26_N9_out11_Mix_before3():
    from src.model.my_v26_N9_out10_Mix_before3 import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset'])
    setting(debug_mode =False,only_test_small_part_dataset=True)
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,setting=setting)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v26_N9_out11_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False, debug_mode=debug_mode)
    # convnet.test(dataloader=cifar10, backup_path='backup/my_v26_N9_out11_Mix_before3/', epoch=50, batch_size=64)

def my_v27_N9_out11_Mix_before3():
    from src.model.my_v27_N9_out11_Mix_before3 import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset'])
    setting = setting(debug_mode =False,only_test_small_part_dataset=True)
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,setting=setting)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v27_N9_out11_Mix_before3/', batch_size=64, n_epoch=50)

    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False, debug_mode=debug_mode)
    # convnet.test(dataloader=cifar10, backup_path='backup/my_v27_N9_out11_Mix_before3/', epoch=50, batch_size=64)

def my_v28_N9_out12_Mix():
    from src.model.my_v28_N9_out12_Mix import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset'])

    # setting = setting(debug_mode =False,only_test_small_part_dataset=True)
    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,setting=setting)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v28_N9_out12_Mix/', batch_size=64, n_epoch=50)

    setting = setting(debug_mode =False,only_test_small_part_dataset=False)
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False,setting=setting)
    convnet.test(dataloader=cifar10, backup_path='backup/my_v28_N9_out12_Mix/', epoch=50, batch_size=64)

def my_v29_N9_out12_Improve1():
    from src.model.my_v29_N9_out12_Improve1 import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset'])

    # setting = setting(debug_mode =False,only_test_small_part_dataset=True)
    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,setting=setting)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v29_N9_out12_Improve1/', batch_size=64, n_epoch=50)

    setting = setting(debug_mode =False,only_test_small_part_dataset=False)
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False,setting=setting)
    convnet.test(dataloader=cifar10, backup_path='backup/my_v29_N9_out12_Improve1/', epoch=30, batch_size=64)

def my_v30_N9_out12_Improve2():
    from src.model.my_v30_N9_out12_Improve2 import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch'])

    # setting = setting(debug_mode =False,only_test_small_part_dataset=True,
    #                   start_n_epoch = 0,test_proprotion=0.9)
    # convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32,is_training =True,setting=setting)
    # convnet.train(dataloader=cifar10, backup_path='backup/my_v30_N9_out12_Improve2_train_second/', batch_size=64, n_epoch=150)

    setting = setting(debug_mode =False,only_test_small_part_dataset=False,
                  start_n_epoch = 51,test_proprotion=0.9)
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, n_layers=32, is_training=False,setting=setting)
    convnet.test(dataloader=cifar10, backup_path='backup/my_v30_N9_out12_Improve2_train_second/', epoch=150, batch_size=64)


# my_cnn()
# residual_net_L20()
# my_residual_v1_N3L20()
# my_residual_v2_N4L20()
# my_residual_v3_N5L20()
# my_residual_v5_N3L44()
# my_residual_v6_RES1_CON14_CON8()
# my_v7_N1_squeeze()
# my_v8_N5_squeeze()
# my_v9_N3_squeeze()
# my_v10_N3_plain_cnn_L14()
# my_v11_N1_plain_cnn_L14()
# my_v12_N3_reidual_L44()
# my_v13_N1_reidual_L44()
# my_v14_N5_reidual_L44()
# my_v15_N1_reidual_L32_google_v2()
# my_v16_N1_reidual_L32_google_v3()
# my_v17_N1_reidual_L32_plain_L3()
# my_v18_N3_Mix_before3()
# my_v19_N9_out9_Mix_before3()
# my_v21_N9_out6_Mix_before3()
# my_v22_N9_out7_Mix_before3()
# my_v23_N9_out8_Mix_before3()
# my_v24_N9_out9_Mix_before3_S()
# my_v25_N9_out9_Mix_before3_ST2()
# my_v26_N9_out11_Mix_before3()
# my_v27_N9_out11_Mix_before3()
# my_v28_N9_out12_Mix()
# my_v29_N9_out12_Improve1()
my_v30_N9_out12_Improve2()













