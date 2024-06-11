import cv2
import os
import numpy as np
# /home/lizhonghao/ALIKE/mgp_valres/epoch1_4_scoremp.png
imgs_filename1 = []
imgs_filename2 = []
base_loc = '/home/lizhonghao/ALIKE/mgp_valres/epoch'
for i in range(1,200):
    imgs_filename1.append(base_loc+f'{i}_6_scoremp.png')
    imgs_filename2.append(base_loc+f'{i}_6.png')
# 设置输入 PNG 图像文件夹路径和视频输出路径
# png_folder = 'input_pngs'
video_name = 'output_video.mp4'

# 获取图像文件夹中的所有 PNG 图像文件名并排序
# images = [img for img in os.listdir(png_folder) if img.endswith(".png")]
# images.sort()

# 获取第一张 PNG 图像的尺寸
frame1 = cv2.imread(imgs_filename1[0])
frame2 = cv2.imread(imgs_filename2[0])
frame = np.concatenate([frame1,frame2],axis=1)
height, width, layers = frame.shape

# 指定输出视频的编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建视频写入器对象
video = cv2.VideoWriter(video_name, fourcc, 5, (width, height))

# 逐帧写入 PNG 图像到视频
for image1,image2 in zip(imgs_filename1,imgs_filename2):
    frame1 = cv2.imread(image1)
    frame2 = cv2.imread(image2)
    frame = np.concatenate([frame1,frame2],axis=1)
    video.write(frame)

# 关闭视频写入器
video.release()
