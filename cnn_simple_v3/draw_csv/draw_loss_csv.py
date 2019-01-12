import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
import os
import numpy as np
import collections
###################### 改这里   ######################
csv_path = 'csv_file/asl_loss/'   #画出CSV文件夹下所有的CSV文件或指定画哪个CSV文件
csv_values = 'Value'   #读取CSV文件的哪一列
compress_number = 50  #10000个数据压缩成125个
axis_x = 150    #显示的x轴坐标范围
#####################   end     #######################

def plot_curve(x,y_datas_dict,y_datas_legend_dict = None,setting_dict={}):
    colors=['r','k','y','c','m','g','b']
    line_styles= ['^-','+-','x-',':','o','*','s','D','.']
    # plt.switch_backend('agg')
    plt.title(setting_dict['title'])
    plt.xlabel(setting_dict['xlabel'])
    plt.ylabel(setting_dict['ylabel'])
    p_legend = []
    p_legend_name = []
    y_datas_keys = y_datas_dict.keys()
    for idx,y_datas_key in enumerate(y_datas_keys):
        y_data_dict = y_datas_dict[y_datas_key]
        p, =plt.plot(x, y_data_dict, line_styles[idx], color=colors[idx],scaley=0.3)
        p_legend.append(p)
        if y_datas_legend_dict is not None:
            p_legend_name.append(y_datas_legend_dict[y_datas_key])
        else:
            p_legend_name.append(y_datas_key)

    plt.legend(p_legend, p_legend_name, loc='center right')

    plt.grid()
    plt.savefig(setting_dict['save_name'], dpi=50, format='png')
    plt.show()

def read_csv(csv_path):
    #如果是文件夹，则把所有的csv文件读取
    if os.path.isdir(csv_path):
        file_list = os.listdir(csv_path)
        for file in file_list:
            data = pd.read_csv(os.path.join(csv_path, file))
            filename = file.split('.')[0]
            yield filename, data
    else:
        # 如果是csv文件，读取这个csv文件
        filename = csv_path.split('/')[-1]
        filename = filename.split('.')[0]
        data = pd.read_csv(os.path.join(csv_path))
        yield filename, data

def compress_data(data,compress_number):
    #必须能整除，如1000个数除以125段
    divide_time = int(len(data)//compress_number)
    new_data = []
    part= []
    for i in range(len(data)):
        part.append(data[i]) #慢慢存够divide_time个数据
        if (i+1) % divide_time ==0:#每divide_time个做一组取平均值
            new_data.append(np.mean(part))
            part=[] #清空

    return new_data
def interpolate_data(y, compress_number=compress_number):
    # 数据插值法,由大量数据压缩成小量数据
    x = range(0, len(y))
    func = interpolate.interp1d(x, y, kind='zero')
    x = np.arange(0, compress_number, 1)
    y = func(x)
    return x, y
if __name__ == "__main__":
    setting_dict = collections.OrderedDict()
    data_dict = collections.OrderedDict()
    # file_list = read_csv(csv_path)
    # 1、读文件，获取数据
    for filename,data in read_csv(csv_path):
        # 从panda长枪中取出数据
        y = data[csv_values].values

        # 2、处理数据：压缩曲线（也可以用数据插值法，这里用自己写的平均点法似乎更准）
        # 数据轴由1000个点压缩成125个
        # y = interpolate_data(y, compress_number=compress_number)
        y = compress_data(data=y, compress_number=compress_number)

        # 3、处理数据：扩张数据（神经网络数据拟合），数据轴125个扩张成150

        # free= 3
        # y = np.polyfit(x, y, 5)
        # x = range(0, len(y))

        # 4、最终取得的数据
        from numpy import arange
        # range函数用来产生一个范围内的整数数组，输入浮点数会出错。因此用arange
        x = arange(0,axis_x,axis_x/len(y))
        data_dict[filename] = y

    # 3、画图
    #标题、x轴、y轴显示信息
    setting_dict['title'] = 'loss curve'
    setting_dict['xlabel'] = 'epoch'
    setting_dict['ylabel'] = 'loss'
    setting_dict['save_name'] ='compare.png'
    plot_curve(x,data_dict,None,setting_dict)































