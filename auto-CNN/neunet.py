# -*- coding: utf8 -*-
# author: ronniecao
import os

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

    setting = setting(debug_mode =True,only_test_small_part_dataset=True,
                  start_n_epoch = 0,test_proprotion=0.94,batch_size=64)

    convnet = ConvNet(n_channel=3, n_classes=10, image_size=24, network_path='src/config/networks/basic.yaml',setting=setting)
    convnet.train(dataloader=cifar10, backup_path='backups/my_v2_plain_TW/', n_epoch=500)
    # convnet.test(backup_path='backup/cifar10-v4/', epoch=0, batch_size=128)


# basic_cnn()
# my_v1_resnet_TW()
my_v2_plain_TW()