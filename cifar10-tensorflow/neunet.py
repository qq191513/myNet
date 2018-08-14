# -*- coding: utf8 -*-
# author: ronniecao
import os
from src.data.cifar10 import Corpus
from keras import backend as K
K.clear_session()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
    convnet.train(dataloader=cifar10, backup_path='backup/my_v10_N3_plain_cnn_L14/', batch_size=64, n_epoch=200)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v10_N3_plain_cnn_L14/', epoch=200, batch_size=64)

def my_v11_N1_plain_cnn_L14():
    from src.model.my_v11_N1_plain_cnn_L14 import ConvNet
    convnet = ConvNet(n_channel=3, n_classes=10, image_size=32, n_layers=44)
    convnet.train(dataloader=cifar10, backup_path='backup/my_v11_N1_plain_cnn_L14/', batch_size=64, n_epoch=200)
    convnet.test(dataloader=cifar10,backup_path='backup/my_v11_N1_plain_cnn_L14/', epoch=200, batch_size=64)


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
my_v11_N1_plain_cnn_L14()