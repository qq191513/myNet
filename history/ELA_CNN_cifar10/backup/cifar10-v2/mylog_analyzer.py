# -*- encoding: utf-8 -*-
# author: ronniecao
import os
import re
import matplotlib.pyplot as plt
import numpy


def load_log(path):
    with open(path, 'r') as fo:
        epoch_list,iter_list,loss_list, valid_precision_list, valid_loss_list = [], [], [],[],[]
        for line in fo:
            line = line.strip()
            pattern = re.compile(
				r'epoch: ([\d]+), iter: ([\d\.]+), train loss: ([\d\.]+), valid precision: ([\d\.]+), valid loss: ([\d\.]+)')

            res = pattern.findall(line)
            if res:
                epoch_list.append(float(res[0][0]))
                iter_list.append(float(res[0][1]))
                loss_list.append(float(res[0][2]))
                valid_precision_list.append(float(res[0][3]))
                valid_loss_list.append(float(res[0][4]))

        return epoch_list, iter_list, loss_list,valid_precision_list,valid_loss_list

def curve_smooth(data_list, batch_size=100):
	new_data_list, idx_list = [], []
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_data_list.append(1.0 * sum(batch) / len(batch))
		idx_list.append(i*batch_size)

	return new_data_list, idx_list

def plot_curve(loss_list,valid_precision_list,epoch_list):
	fig = plt.figure(figsize=(10, 5))
	plt.subplot(121)
	p1 = plt.plot(epoch_list, loss_list, '.--', color='#6495ED')
	plt.grid(True)
	plt.title('cifar10 image classification loss')
	plt.xlabel('# of epoch')
	plt.ylabel('loss')
	plt.subplot(122)
	p2 = plt.plot(epoch_list,valid_precision_list, '.--', color='#66CDAA')
	# p3 = plt.plot(valid_precision_idxs, valid_precision_list, '.--', color='#FF6347')
	# plt.legend(p2[0], 'valid_precision_list')
	plt.grid(True)
	plt.title('cifar10 image classification precision')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	# plt.show()
	plt.savefig('cifar10-v1.png', dpi=72, format='png')


# epoch_list, iter_list, loss_list,valid_precision_list,valid_loss_list = load_log('test_log.txt')
# print(numpy.array(loss_list[-100:]).mean(), numpy.array(iter_list[-100:]).mean())
# loss_list, loss_idxs = curve_smooth(loss_list[0:500], batch_size=1)
# train_precision_list, train_precision_idxs = curve_smooth(iter_list, batch_size=10)
# valid_precision_list, valid_precision_idxs = curve_smooth(valid_precision_list, batch_size=10)
# plot_curve(loss_list, valid_precision_list, epoch_list)

epoch_list, iter_list, loss_list,valid_precision_list,valid_loss_list = load_log('test_log.txt')
plot_curve(loss_list, valid_precision_list, epoch_list)

