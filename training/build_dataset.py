import numpy as np
import cv2
import os
import yaml
import sys
sys.path.append('../')
import magic_point_dataset.synthetic_data as synthetic_data
from datetime import datetime
wdic_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
para_loc =os.path.join(wdic_path, "magic_point_dataset/config.yaml")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import magic_point_dataset.photometric_augmentation as photoaug
from magic_point_dataset.homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points
import tensorflow_addons as tfa
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

#添加噪声
def photometric_augmentation(img, config):
    primitives = config["data"]["augmentation"]["photometric"]["primitives"]
    params = config["data"]["augmentation"]["photometric"]["params"]

    prim_configs = [params.get(p, {}) for p in primitives]
    for j, (p, c) in enumerate(zip(primitives, prim_configs)):
        img = getattr(photoaug, p)(img, **c)

    return img

#对目标物进行单应性变换
def homographic_augmentation(img , points, **config):
    params = config["params"]
    valid_border_margin = config["valid_border_margin"]

    image_shape = np.array(img.shape[:2])
    homography = sample_homography(image_shape, **params)[0]
    warped_image = tfa.image.transform(
        img, homography, interpolation='BILINEAR')
    valid_mask = compute_valid_mask(image_shape, homography, valid_border_margin)

    warped_points = warp_points(points, homography)
    warped_points = filter_points(warped_points, image_shape)

    return warped_image , np.flip(warped_points, 1) , homography , valid_mask





def print_current_time(location,msg):
    # 获取当前时间
    current_time = datetime.now()

    # 将时间格式化为字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # 打印当前时间
    print(formatted_time+' '+location+' '+msg)

# 调用函数打印当前时间


with open(para_loc, "r") as f:
    config = yaml.safe_load(f)
    #location = config['pic_loc']
#生成几何数据集的类型
# draw_ellipses 和 gaussian_noise类别无ground truth特征点
primitives = [
    'draw_lines',
    'draw_polygon',
    'draw_multiple_polygons',
    #'draw_ellipses',
    'draw_star',
    'draw_checkerboard',
    'draw_stripes',
    'draw_cube',
    #'gaussian_noise'
]

#随机生成一张图片（无噪声和变换）
def gen_item():
    primitive = np.random.choice(primitives)
    image = synthetic_data.generate_background(
                    config["data"]['generation']['image_size'], **config["data"]['generation']['params']['generate_background'])
    points = np.array(getattr(synthetic_data, primitive)(
                    image, **config["data"]['generation']['params'].get(primitive, {})))
    
    # magicpt_path = os.path.join(wdic_path,'magic_point_dataset')
    # if self.save_imgs:
    #     path = os.path.join(magicpt_path,config['data']['save_path'],'{}.png'.format(item))
    #     cv2.imwrite(path,image)
    #     path = os.path.join(magicpt_path,config['data']['save_path'],'{}.npy'.format(item))
    #     np.save(path, points)
    return image.astype(np.float32),np.flip(points.astype(np.float32), 1)

#生成图片并储存
def build_dataset(location):
    data_lenth = config['data_lenth']
    for id in range(data_lenth):
        image,points = gen_item()
        image = np.expand_dims(image, axis=-1)
        #data = {"image":image,"points":points}
        if config["data"]["augmentation"]["photometric"]["enable"]:
            image = photometric_augmentation(image,config=config)
        if config["data"]["augmentation"]["homographic"]["enable"]:
            image_trans , warped_points, homography, valid_mask = homographic_augmentation(image, points,**config["data"]["augmentation"]["homographic"])
        points = (points * np.array(config['data']['preprocessing']['resize'], np.float32) / np.array(config['data']['generation']['image_size'], np.float32))
        warped_points = (warped_points.numpy() * np.array(config['data']['preprocessing']['resize'], np.float32) / np.array(config['data']['generation']['image_size'], np.float32))

        image = cv2.resize(image.numpy(), tuple(config['data']['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)
        image_trans = cv2.resize(image_trans.numpy(), tuple(config['data']['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)
        
        #image_trans = image_trans.numpy()
        # points = points.numpy()
        # warped_points = warped_points.numpy()
        homography = homography.numpy()
        H = np.concatenate((homography,np.array([1])),axis = 0).reshape(3,3)
        valid_mask = valid_mask.numpy()
        # points = torch.tensor(points.numpy())
        # homography = torch.tensor(homography.numpy())
        # valid_mask = torch.tensor(valid_mask.numpy())
        h0, w0= image.shape
        h1, w1 = image_trans.shape


        if config['is_graystyle']:
            pass 
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image_trans = cv2.cvtColor(image_trans, cv2.COLOR_GRAY2RGB)   
        loc = location
        pic_name_trans = os.path.join(loc,'trans_{}.png'.format(id))
        pic_name = os.path.join(loc,'ori_{}.png'.format(id))
        pt_name_trans = os.path.join(loc,'pt_trans_{}.npy'.format(id))
        pt_name = os.path.join(loc,'pt_{}.npy'.format(id))
        homography_path = os.path.join(loc,'homo_{}.npy'.format(id))


        cv2.imwrite(pic_name,image)
        cv2.imwrite(pic_name_trans,image_trans)
        np.save(homography_path, H)
        np.save(pt_name,points)
        np.save(pt_name_trans,warped_points)

# 文件dataset_state.txt维护最后更新的数据集，通过重复运行本脚本可以实现在训练中每个epoch输入的图片都不同
def main(): 
    location = sys.argv[1]  
    print_current_time(location,"start build") 
    build_dataset(location)
    print_current_time(location,"end build")
    with open("dataset_state.txt","a") as f:
        f.write("\n" + location + " is ready!")
    
if __name__ == "__main__":
    main()