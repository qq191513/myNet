


if __name__=='__main__':
    choice = 3
    if choice == 1:# 1、训练mnist数据集
        from train.trainer_for_mnist import train_mnist_model
        train_mnist_model()
    elif choice == 2:# 2、训练cifar10数据集
        from train.trainer_for_cifar10 import train_cifar10_model
        train_cifar10_model()
    elif choice == 3: # 3、训练普通图片集
        from train.trainer import train_model
        train_model()
    elif choice == 4: # 4、测试普通图片集
        from eval.evaluate import evaluate_result
        evaluate_result()
    else:
        pass

