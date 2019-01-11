# -*- coding: utf8 -*-
# author: ronniecao
import os
import tensorflow as tf
from collections import namedtuple
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from src.data.cifar10 import Corpus
cifar10 = Corpus()

def basic_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/basic.yaml')
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v1/', batch_size=128, n_epoch=500)
    # convnet.test(dataloader=cifar10, backup_path='backup/cifar10-v2/', epoch=5000, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def vgg_cnn():
    from src.model.basic_cnn import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/vgg.yaml')
    # convnet.debug()
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v2/', batch_size=128, n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v3/', epoch=0, batch_size=128)
    # convnet.observe_salience(batch_size=1, n_channel=3, num_test=10, epoch=2)
    # convnet.observe_hidden_distribution(batch_size=128, n_channel=3, num_test=1, epoch=980)
    
def resnet():
    from src.model.resnet import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/resnet.yaml')
    convnet.train(dataloader=cifar10, backup_path='backups/cifar10-v5/', batch_size=128, n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)

def my_v1_resnet_TW():
    from src.model.my_v1_resnet_TW import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size'])

    setting = setting(debug_mode =True,only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64)

    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/my_resnet.yaml',setting=setting)
    convnet.train(dataloader=cifar10, backup_path='backups/auto_v1_net/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


def my_v2_plain_TW():
    from src.model.my_v2_plain_TW import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size'])

    setting = setting(debug_mode =False,only_test_small_part_dataset=False,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64)

    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/basic.yaml',setting=setting)
    convnet.train(dataloader=cifar10, backup_path='backups/my_v2_plain_TW/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


def my_v3_plain_TW_gen_one_layer():
    from src.model.my_v3_plain_TW_gen_one_layer import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size',
                                    'output_graph'])

    setting = setting(debug_mode =True,only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64,
                      output_graph = True
                      )

    convnet = ConvNet(dataloader=cifar10,n_channel=3, n_classes=10, image_size=24,
                      network_path='src/config/networks/basic.yaml',setting=setting)
    convnet.train(backup_path='backups/my_v3_plain_TW_gen_one_layer/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


def my_v4_plain_TW_gen_one_layer():
    from src.model.my_v4_plain_TW_gen_one_layer import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size',
                                    'output_graph'])

    setting = setting(debug_mode =True,only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64,
                      output_graph = True
                      )

    convnet = ConvNet(dataloader=cifar10,n_channel=3, n_classes=10, image_size=24,
                      network_path='src/config/networks/basic.yaml',setting=setting)
    convnet.train(backup_path='backups/my_v4_plain_TW_gen_one_layer/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


def my_v5_plain_TW_gen_one_layer():
    from src.model.my_v5_plain_TW_gen_one_layer import ConvNet
    setting = namedtuple('setting',['debug_mode','only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size',
                                    'output_graph'])

    setting = setting(debug_mode =True,only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64,
                      output_graph = True
                      )

    convnet = ConvNet(dataloader=cifar10,n_channel=3, n_classes=10, image_size=24,
                      network_path='src/config/networks/basic.yaml',setting=setting,
                        scope_name = 'first')
    convnet.train(backup_path='backups/my_v5_plain_TW_gen_one_layer/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)



def my_v6_plain_TW():
    from src.model.my_v6_plain_double import ConvNet
    from src.model.my_v6_plain_double import train_obeject

    # 构建会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    train_setting = namedtuple('train_setting',['only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size','n_epoch',
                                    'backup_path',])
    train_setting = train_setting(only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64,n_epoch= 2,
                backup_path='backups/my_v6_plain_TW/')

    net_dict_first = ConvNet(network_path='src/config/networks/basic.yaml',n_channel=3, n_classes=10, image_size=24,
                       name_scope='first')
    train_obeject(dataloader=cifar10, setting = train_setting,net_dict=net_dict_first,sess=sess)

    net_dict_second = ConvNet(network_path='src/config/networks/basic.yaml', n_channel=3, n_classes=10, image_size=24,
                       name_scope='second')
    train_obeject(dataloader=cifar10, setting = train_setting,net_dict=net_dict_second,sess=sess)

    tf.summary.FileWriter(train_setting.backup_path,sess.graph)
    sess.close()
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)




def my_v7_plain_double_concact_to_one():
    from src.model.my_v7_plain_double_concact_to_one import ConvNet
    from src.model.my_v7_plain_double_concact_to_one import Conv_new_Net
    from src.model.my_v7_plain_double_concact_to_one import train_obeject
    from src.model.my_v7_plain_double_concact_to_one import merge_weight

    # 构建会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    train_setting = namedtuple('train_setting',['only_test_small_part_dataset',
                                    'test_proprotion','start_n_epoch','batch_size','n_epoch',
                                    'backup_path',])
    train_setting = train_setting(only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.97,batch_size=64,n_epoch= 1,
                backup_path='backups/my_v7_plain_double_concact_to_one/')

    net_dict_first = ConvNet(network_path='src/config/networks/basic.yaml',n_channel=3, n_classes=10, image_size=24,
                       name_scope='first')
    train_obeject(dataloader=cifar10, setting = train_setting,net_dict=net_dict_first,sess=sess)




    net_dict_second = ConvNet(network_path='src/config/networks/basic.yaml', n_channel=3, n_classes=10, image_size=24,
                       name_scope='second')
    train_obeject(dataloader=cifar10, setting = train_setting,net_dict=net_dict_second,sess=sess)

    # 取出权重合并
    var = tf.global_variables()
    first_weight = [val for val in var if 'kernel:0' in val.name and 'first' in val.name and 'conv' in val.name]
    second_weight = [val for val in var if 'kernel:0' in val.name and 'second' in val.name and 'conv' in val.name]

    weight_dict= merge_weight(first_weight,second_weight)
    # for name,value in weight_dict.items():
        # print(name)
        # print(sess.run(value))

    net_dict_first = Conv_new_Net(sess=sess,network_path='src/config/networks/basic.yaml', n_channel=3, n_classes=10, image_size=24,
                             name_scope='merge_net',weight_dict=weight_dict)
    train_obeject(dataloader=cifar10, setting=train_setting, net_dict=net_dict_first, sess=sess)



    tf.summary.FileWriter(train_setting.backup_path,sess.graph)
    sess.close()
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)






# basic_cnn()
# my_v1_resnet_TW()
# my_v2_plain_TW()
# my_v3_plain_TW_gen_one_layer()
# my_v4_plain_TW_gen_one_layer()
# my_v5_plain_TW_gen_one_layer()
# my_v6_plain_TW()
my_v7_plain_double_concact_to_one()


