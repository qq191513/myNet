import os
import random

def single_folder_rename(where,new_name,new_postfix):  #仅对单个文件夹里面的所有文件命名
    files =os.listdir(where)  #列出路径where下的每个文件夹
    for index,filename in enumerate(files):
        final_name = new_name + '_{index}'+ new_postfix
        final_name_1 = final_name.format(index=index)
        final_name_2 = os.path.join(where,final_name_1)
        if os.path.exists(final_name_2):  # 若名字重复则在newname前面加上xxx_random_number
            random_number = random.random()
            random_number = int(random_number * 100000)
            final_name_1 = 'xxx_{random_number}_' + final_name_1
            final_name_1 = final_name_1.format(random_number = random_number)
            final_name_2 = os.path.join(where, final_name_1)
            print('name conflict!!! add xxx_random_number to name')
        print('new name is : ' + final_name_2)
        old_name = os.path.join(where,filename)
        os.rename(old_name,final_name_2)# 修改
    print('single_folder_rename work done !')



def double_folders_rename(where, new_name, new_postfix):  #对双层文件夹里面的所有文件命名（即对某个文件夹里面的多个文件夹里面各自的文件重命名）
    folders =os.listdir(where)  #列出路径where下的每个文件夹
    for index,foldername in enumerate(folders):
        print('{index} dealing folder :{foldername}'.format(index = index,foldername =foldername))
        single_folder_rename(os.path.join(where,foldername), new_name, new_postfix)


where = 'datasets/dataset3/validation/'
new_name = 'validation'
new_postfix = '.jpg'
double_folders_rename(where,new_name,new_postfix)

