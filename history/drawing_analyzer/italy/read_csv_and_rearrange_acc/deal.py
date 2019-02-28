#本文件专用处理tensorboard生成的csv数据文件，保存到新的txt文件

import csv
import numpy as np
####################   改这里  #############################
read_name = 'em_italy-train_acc.csv'
save_name = 'em_italy-train_acc_150.txt'
divide_number = 150  #数据压缩成多少个
new_axis_name ='x_axis' #根据数据长度生成一个统一的x轴画图用,不需要则填None
int_Value = 'Step'  #想转换成整数的数据(列)，用于做横坐标，不需要则填None
del_col = ['Wall time'] #不需要的数据(列),不需要则填None
save_csv_style = False #到底保存成csv格式（表格）还是txt格式（字典格式）
####################   end       ###########################

def read_csv(filename =None):
    with open(filename) as f:
        ori_data = []
        dict_reader = csv.DictReader(f)  #字典式读入

        #删除没什么卵用的数据并显示数据
        print('原始数据：')
        for each in dict_reader:
            for i in del_col:
                print(each)
                del each[i]
            ori_data.append(each)  #读取出来装到内存再说
    return ori_data

def get_interval(data=None,divide_number=None):
    total_number = len(ori_data)  #总个数
    batch_number = total_number//divide_number  #每批多少个
    # 生成划分区间 divide_interval
    divide_interval = []
    for i in range(divide_number-1):
        divide_interval.append((i*batch_number,(i+1)*batch_number))
    i = i + 1
    divide_interval.append((i*batch_number,total_number))  #若凑不成整数，最后一批特殊数量
    return divide_interval

def compress_data_value(ori_data=None,divide_interval=None,is_add_increasing_number_as_x_axis=None):
    new_data = []
    new_axis_number = 0
    for start,end in divide_interval:   #取区间范围
        sum = {} #每个区间累加前清空
        for i in range(start,end):
            get_dict = ori_data[i] #取出每个字典
            for key,value in get_dict.items():
                if not key in sum:
                    sum[key] = 0
                sum[key] += np.float(value)

        for key,value in sum.items():  #对每一个键值对mean
            sum[key] = sum[key] / (end - start)
            if int_Value == key:
                sum[key] = int(sum[key])
        if new_axis_name is not None:#添加新轴
            sum[new_axis_name]=new_axis_number
            new_axis_number +=1

        new_data.append(sum)  #放进新数组
    return new_data


def save_csv(save_file = None,save_data=None):
    #取列名
    cols_name = []
    tmp = save_data[0]
    for key, value in tmp.items():
        cols_name.append(key)

    with open(save_file, 'w') as csvFile:
        #先保存列名到第一行
        writer = csv.writer(csvFile)
        writer.writerow(cols_name)

        #再根据列名取出字典做为一行数据来保存
        for data_dict in save_data:
            print(data_dict)
            save_data_row = []
            for col_name in cols_name:
                save_data_row.append(data_dict[col_name])
            writer.writerow(save_data_row)

def save_txt(save_file = None,save_data=None):
    with open(save_file, 'w') as txtFile:
        for row in save_data:
            row = str(row)
            row= row + '\r\n'
            txtFile.write(row)



ori_data = read_csv(read_name)
divide_interval = get_interval(data=ori_data, divide_number=divide_number)
new_data = compress_data_value(ori_data=ori_data,divide_interval=divide_interval)


print('#############################################################################################################################')
print('#############################################################################################################################')
print('########## new ##########')
print('#############################################################################################################################')
print('#############################################################################################################################')
if save_csv_style:
    save_csv(save_file = save_name, save_data = new_data)
else:
    save_txt(save_file = save_name, save_data = new_data)

print('save done!')


