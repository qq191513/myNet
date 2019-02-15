import collections
import sys
sys.path.append('../')
import tools.development_kit as dk
import os
chinese_ttf=None
setting_dict = collections.OrderedDict()
###################### asl 改这里   ######################
csv_path = 'csv_file/asl_acc/'   #画出CSV文件夹下所有的CSV文件或指定画哪个CSV文件
csv_values = 'Value'   #读取CSV文件的哪一列
compress_number = 50  #10000个数据压缩成125个
axis_x = 150    #显示的x轴坐标范围
chinse_show =False
setting_dict['title'] = 'ASL dataset classification precision'
setting_dict['xlabel'] = 'epoch'
setting_dict['ylabel'] = 'accuracy'
#####################   end     #######################

###################### italy 改这里   ######################
# csv_path = 'csv_file/italy_acc/'   #画出CSV文件夹下所有的CSV文件或指定画哪个CSV文件
# csv_values = 'Value'   #读取CSV文件的哪一列
# compress_number = 50  #10000个数据压缩成125个
# axis_x = 150    #显示的x轴坐标范围
# setting_dict['title'] = 'ISL dataset classification precision'
# chinse_show =False
# setting_dict['xlabel'] = 'epoch'
# setting_dict['ylabel'] = 'accuracy'
#####################   end     #######################



###################### italy chinse 改这里   ######################
# csv_path = 'csv_file/italy_acc_chinese/'   #画出CSV文件夹下所有的CSV文件或指定画哪个CSV文件
# csv_values = 'Value'   #读取CSV文件的哪一列
# compress_number = 50  #10000个数据压缩成125个
# axis_x = 150    #显示的x轴坐标范围
# setting_dict['title'] = '不同算法的手势识别正确率比较'
# setting_dict['xlabel'] = '训练轮数'
# setting_dict['ylabel'] = '正确率'

# chinese_ttf = os.path.join('/home/mo/tool/font/SIMKAI.TTF')
# chinse_show =True
#####################   end     #######################

if __name__ == "__main__":

    data_dict = collections.OrderedDict()
    # 1、读文件，获取数据
    for filename,data in dk.read_csv(csv_path):
        # 1、从panda长枪中取出数据
        y = data[csv_values].values

        # 2、处理数据：压缩曲线（也可以用数据插值法，这里用自己写的平均点法似乎更准）
        # 数据轴由1000个点压缩成125个
        # y = interpolate_data(y, compress_number=compress_number)
        y = dk.compress_data(data=y, compress_number=compress_number)

        # 3、最终取得的数据
        from numpy import arange
        # range函数用来产生一个范围内的整数数组，输入浮点数会出错。因此用arange
        x = arange(0,axis_x,axis_x/len(y))
        data_dict[filename] = y

    # 4、画图
    #标题、x轴、y轴显示信息
    setting_dict['save_name'] = 'compare.png'
    #用英文显示
    if chinse_show:
        # 用中文显示
        dk.plot_curve_chinese_font(x, data_dict, None, setting_dict, chinese_ttf)
    else:
        # 用英文显示
        dk.plot_curve(x,data_dict,None,setting_dict)
































