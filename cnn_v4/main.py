


if __name__=='__main__':
    choice = 2
    if choice == 1:
        from train.train_cnn_mnist import train_mnist_model
        train_mnist_model()
    elif choice == 2:
        from train.train_cnn_cifar10 import train_cifar10_model
        train_cifar10_model()
    elif choice == 3:
        from train.train_cnn_asl_isl import train_asl_isl_model
        train_asl_isl_model()
    else:
        pass