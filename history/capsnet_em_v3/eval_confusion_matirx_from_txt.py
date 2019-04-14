import tensorflow as tf
import config as cfg
from sklearn.metrics import confusion_matrix

#用法介绍
# labels=["ant", "bird", "cat"]
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# C=confusion_matrix(y_true, y_pred,labels=labels)
# print(C)

from confusion_matrix_API import plot_confusion
from confusion_matrix_API import plot_confusion_chinese
import os
####################   改这里  ##########################################
# dataset_name = 'asl'
# recognize_data_dir ='../data/asl_tf'
# recognize_labels_txt_keywords = 'labels.txt'
#
# predicts_and_labels_txt = 'predicts_and_labels.txt'
# title='Normalized confusion matrix'
chinese_show = False
####################   end       ########################################

####################   中文显示 改这里  ##########################################
dataset_name = 'asl'
recognize_data_dir ='../data/asl_tf'
recognize_labels_txt_keywords = 'labels.txt'
predicts_and_labels_txt = 'predicts_and_labels.txt'
title='Normalized confusion matrix'
chinese_show = True
chinese_ttf = os.path.join('/home/mo/tool/font/SIMKAI.TTF')
save_dir = '/home/mo/work/output/capsnet_em_v3'
####################   end       ########################################
os.makedirs(save_dir,exist_ok=True)
def conver_number_to_label_name(the_list,labels_maps):
    for idx,each in enumerate(the_list):
        the_list[idx] = labels_maps[str(each)]
    return the_list

def main(args):
    font_dict = {
        'xlabel': 10, 'ylabel': 10,
        'xticklabels': 6, 'yticklabels': 6,
        'rate_fontsize': 5.3
    }
    labels_txt = cfg.search_keyword_files(recognize_data_dir, recognize_labels_txt_keywords)
    labels_maps = cfg.read_label_txt_to_dict(labels_txt[0])
    np_predicts_list = []
    np_lables_list = []

    with open(predicts_and_labels_txt, 'r') as f:
        line = f.readline()
        while line:
            if 'labels' in line:
                line = f.readline()
                line = line.replace('[','')
                line = line.replace(']','')
                line = line.split(',')
                line = list(map(int,line))
                np_lables_list = line
            if 'predicts' in line:
                line = f.readline()
                line = line.replace('[','')
                line = line.replace(']','')
                line = line.split(',')
                line = list(map(int,line))
                np_predicts_list =line
            line = f.readline()


        # 7、根据step 6求得混淆矩阵
        labels = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f',
                  'g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        # labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
        #           '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']

        np_predicts_list = conver_number_to_label_name(np_predicts_list,labels_maps)
        np_lables_list = conver_number_to_label_name(np_lables_list,labels_maps)

        cm=confusion_matrix(y_true=np_lables_list, y_pred=np_predicts_list)
        print(cm)
        # cmap = 'Accent_r'

        all_color = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',\
        'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', \
        'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',\
        'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',\
        'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',\
        'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',\
        'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', \
        'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',\
        'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r',\
        'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', \
        'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cool', \
        'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', \
        'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',\
        'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', \
        'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',\
        'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', \
        'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', \
        'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean',\
        'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', \
        'rainbow_r', 'seismic', 'seismic_r', 'spectral', 'spectral_r', 'spring', 'spring_r',\
        'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', \
        'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'viridis', 'viridis_r', 'winter', 'winter_r']

        all_color = ['hot']
        for cmap in  all_color:
            savefig =cmap + '_confusion_matrix.png'
            savefig = os.path.join(save_dir,savefig)
            try:
                if chinese_show:
                    plot_confusion_chinese(cm, title=title, labels=labels, cmap=cmap,
                                   savefig=savefig, font_dict=font_dict,chinese_ttf = chinese_ttf)
                else:
                    plot_confusion(cm, title=title,labels =labels ,cmap=cmap,
                        savefig=savefig,font_dict=font_dict )
            except Exception:
                print(cmap,' is unvalid ! ')






if __name__ == "__main__":
    tf.app.run()










