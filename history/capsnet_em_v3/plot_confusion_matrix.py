#coding:utf-8
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



def plot_confusion(cm, title='Confusion Matrix',labels= None,cmap=None,savefig=None,font_dict=None):
    # 创建图
    plt.figure(figsize=(9, 4), dpi=150) #例如figsize=(5, 5)相当于生成一张500*500大小的图片， 调试的时候用低分辨dpi=150，应用的时候再用高分辨率1500
    np.set_printoptions(precision=2)  #设置小数点保留精度

    # 按行归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    # # 不归一化
    # cm_normalized = cm
    # 网格
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    # 写字
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=font_dict['rate_fontsize'], va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.2f" % (0.00,), color='white', fontsize=font_dict['rate_fontsize'], va='center',ha='center')
    # offset the tick
    tick_marks = np.array(range(len(labels))) + 0.5

    # 获取当前子图Get Current Axes
    # plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_xticklabels(tick_marks, fontdict={'fontsize':font_dict['yticklabels']}, minor=False)
    # plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().set_yticklabels(tick_marks, fontdict={'fontsize':font_dict['yticklabels']}, minor=False)
    plt.gca().xaxis.set_ticks_position('top')  #将xaxis ticks设置在顶部
    plt.gca().yaxis.set_ticks_position('none')
    # 生成网格
    # plt.grid(True, which='minor', linestyle='-')
    plt.grid(True, which='minor', linestyle='-')
    # 获取当前的图表（Get Current Figure）
    plt.gcf().subplots_adjust(bottom=0.15)


    # 标题
    # plt.title(title,fontdict={'fontsize':30})
    # 柱条
    # plt.colorbar()
    # x、y轴显示
    xlocations = np.array(range(len(labels)))
    # 显示xticks
    # plt.xticks(xlocations, labels, rotation=90)
    plt.xticks(xlocations, labels)
    # 显示yticks
    plt.yticks(xlocations, labels)
    # 显示ylabel
    plt.ylabel('True label',fontdict={'fontsize':font_dict['ylabel']})
    # 显示xlabel
    plt.xlabel('Predict',fontdict={'fontsize':font_dict['xlabel']})
    # 显示图 (aspect =0.5可以画成长方形，因为规模太大的话正方形根本看不清)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect=0.45)
    # 保存图
    plt.savefig(savefig, format='png')
    # 显示图
    plt.show()

if __name__ == "__main__":
    #使用样板
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    labels=["ant", "bird", "cat"]
    cm = confusion_matrix(y_true, y_pred)
    # cmap = plt.cm.binary
    cmap = 'Accent_r'
    font_dict={
        'xlabel':25,'ylabel':25,
        'xticklabels':15,'yticklabels':15,
        'rate_fontsize':25
    }
    plot_confusion(cm, title='Normalized confusion matrix',
                labels =labels ,cmap=cmap,savefig='confusion_matrix.png',font_dict=font_dict)



# cmap的候选值有以下颜色选择：
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',\
# 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', \
# 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',\
# 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',\
# 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',\
# 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',\
# 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', \
# 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Vega10',\
# 'Vega10_r', 'Vega20', 'Vega20_r', 'Vega20b', 'Vega20b_r', 'Vega20c', 'Vega20c_r', \
# 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',\
# 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', \
# 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cool', \
# 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', \
# 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',\
# 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', \
# 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',\
# 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', \
# 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', \
# 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',\
# 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', \
# 'rainbow_r', 'seismic', 'seismic_r', 'spectral', 'spectral_r', 'spring', 'spring_r',\
# 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', \
# 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

