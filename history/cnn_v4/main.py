


if __name__=='__main__':
    choice = 1
    if choice == 1:# 1、cnn模型 训练mnist数据集
        from train.train_cnn_mnist import train_mnist_model
        train_mnist_model()
    elif choice == 2:# 2、cnn模型 训练cifar10数据集
        from train.train_cnn_cifar10 import train_cifar10_model
        train_cifar10_model()
    elif choice == 3: # 3、cnn模型 训练asl或isl数据集
        from train.train_cnn_asl_isl import train_asl_isl_model
        train_asl_isl_model()
    else:
        pass