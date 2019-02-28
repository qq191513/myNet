from PIL import Image, ImageDraw, ImageFont

import os
import time
# image_path = '/home/mo/work/caps_face/Matrix-Capsules-EM-Tensorflow-master/data/group'  # 图片集地址
image_path = 'g3'  # 图片集地址
image_format = ['.jpg', '.JPG', '.png']  # 图片格式
image_size = 256  # 每张小图片的大小
image_row = 1  # 图片间隔，也就是合并成一张图后，一共有几行
image_column = 5  # 图片间隔，也就是合并成一张图后，一共有几列
image_save_path = 'final.jpg'  # 图片转换后的地址
# ttfont = ImageFont.truetype('simhei.ttf', 10)  # 字体大小
# font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体


# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(image_path) for item in image_format if
               os.path.splitext(name)[1] == item]
image_names.sort()
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != image_row * image_column:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, image_row + 1):
        for x in range(1, image_column + 1):
            img = Image.open(os.path.join(image_path,image_names[image_column * (y - 1) + x - 1])).resize(
                (image_size, image_size), Image.ANTIALIAS)
            label = image_names[image_column * (y - 1) + x - 1].split('.')[0]
            # 1、 先让图片左上角写个标注
            # from_image = ImageDraw.Draw(img)  # 修改图片
            # from_image.text((2, 2), label,fill=(255,255,255))  # 利用ImageDraw的内置函数，在图片上写入文字
            # img.show()
            # img.save(os.path.join('new',label+'.jpg'))
            # 2、 保存成一张大图
            to_image.paste(img, ((x - 1) * image_size, (y - 1) * image_size))

    to_image.show()  #Image的显示像cv2一样要打开Xming才行
    to_image.save(image_save_path)  # 保存新图
    time.sleep(10)


image_compose()  # 调用函数
