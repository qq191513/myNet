# -*- encoding: utf-8 -*-
# author: mo weilong
import re
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_parameters_ISLt(path,re_dict={}):
	data_dict =OrderedDict()
	for key in re_dict.keys():
		with open(path, 'r') as fo:
			number_ISLt= []
			for line in fo:
				pattern = re.compile(re_dict[key])
				res = pattern.findall(line)
				if res:
					number_ISLt.append(float(res[0]))
		data_dict[key] =number_ISLt
	return data_dict

def curve_smooth(data_ISLt, batch_size=100):
	new_data_ISLt, idx_ISLt = [], []
	for i in range(int(len(data_ISLt) / batch_size)):
		batch = data_ISLt[i*batch_size: (i+1)*batch_size]
		new_data_ISLt.append(1.0 * sum(batch) / len(batch))
		idx_ISLt.append(i*batch_size)

	return new_data_ISLt, idx_ISLt

def plot_curvev_v2(x,y_datas_dict,y_datas_legend_dict = None,setting_dict={}):
    colors=['r','k','y','c','m','g','b']
    line_styles= ['*-','-','-o','x','^',':','s','D','.','+']
    # plt.switch_backend('agg')
    plt.title(setting_dict['title'])
    plt.xlabel(setting_dict['xlabel'])
    plt.ylabel(setting_dict['ylabel'])
    p_legend = []
    p_legend_name = []
    y_datas_keys = y_datas_dict.keys()
    for idx,y_datas_key in enumerate(y_datas_keys):
        y_data_dict = y_datas_dict[y_datas_key]
        p, =plt.plot(x, y_data_dict, line_styles[idx], color=colors[idx])
        p_legend.append(p)
        if y_datas_legend_dict is not None:
            p_legend_name.append(y_datas_legend_dict[y_datas_key])

    if p_legend_name is not None:
        plt.legend(p_legend, p_legend_name, loc='center right')

    plt.grid()
    plt.savefig(setting_dict['save_name'], dpi=100, format='png')
    plt.show()
    print('done!')


if __name__ =='__main__':
    file_path = 'em_italy-all_loss_150.txt'
    file_path_1 = 'em_v3_italy-all_loss_150.txt'
    # file_path_2 = 'capsnet_em_v3-all_loss.txt'


    # 从文件中正则re获取全部y轴的值
    #file_path_1
    y_re_dict_1 = OrderedDict()
    y_re_dict_1['capsnet_Value']=r'\'Value\': ([\d\.]+)'
    y_datas_dict_1 = get_parameters_ISLt(path = file_path ,re_dict= y_re_dict_1)

    #file_path_2
    y_re_dict_2 = OrderedDict()
    y_re_dict_2['capsnet_v1_Value']=r'\'Value\': ([\d\.]+)'
    y_datas_dict_2 = get_parameters_ISLt(path = file_path_1 ,re_dict= y_re_dict_2)

    #file_path_3
    # y_re_dict_3 = OrderedDict()
    # y_re_dict_3['capsnet_v3_Value']=r'\'Value\': ([\d\.]+)'
    # y_datas_dict_3 = get_parameters_ISLt(path = file_path_2 ,re_dict= y_re_dict_3)

    #两个顺序字典合并
    y_datas_dict = OrderedDict()
    y_datas_dict.update(y_datas_dict_1)
    y_datas_dict.update(y_datas_dict_2)
    # y_datas_dict.update(y_datas_dict_3)

    #从文件中正则re获取全部x轴的值
    x_re_dict = OrderedDict()
    x_re_dict['x_axis']=r'\'x_axis\': ([\d\.]+)'

    x_datas_dict = get_parameters_ISLt(path = file_path ,re_dict= x_re_dict)

    #画图显示legend的名字
    y_datas_legend_dict =OrderedDict()
    y_datas_legend_dict['capsnet_Value']="Caps-em Original"
    y_datas_legend_dict['capsnet_v1_Value']="Caps-em improved"
    # y_datas_legend_dict['capsnet_v3_Value']="capsnet_em_v3"


    #标题、x轴、y轴显示信息
    setting_dict = OrderedDict()
    setting_dict['title'] = 'The loss value of the Caps-em in ISL dataset'
    setting_dict['xlabel'] = 'x_axis'
    setting_dict['ylabel'] = 'accuracy'
    setting_dict['save_name'] ='capsuleNet-EM_final_v3.png'

    #传入字典参数并画图
    plot_curvev_v2(x_datas_dict['x_axis'],y_datas_dict,y_datas_legend_dict,setting_dict)


