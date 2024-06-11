import sys
sys.path.append('../')
import yaml
import numpy as np
import torch
import cv2
import os
wdic_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
para_loc =os.path.join(wdic_path, "magic_point_dataset/config.yaml")
with open(para_loc, "r") as f:
    config = yaml.safe_load(f)


def add_keypoint_map(img, points ,gauss_matrix,rad = 30,is_deeptype=False, add_gauss =False ,cell_size = 8):
    assert (is_deeptype and add_gauss) != True
    img = torch.tensor(img)
    image_shape = img.shape[:2]
    points = torch.tensor(points)
    kp = torch.minimum(points.to(torch.int32), torch.tensor(image_shape)-1)
    w, h = image_shape[0]//cell_size,image_shape[1]//cell_size
    if is_deeptype:
        jud = torch.ones((w, h),dtype=torch.int32)
        kmap = torch.zeros((w,h,cell_size**2))
        for id in range(kp.shape[0]):
            row_id = kp[id,0]//cell_size
            col_id = kp[id,1]//cell_size
            cell_id = (kp[id,0]%cell_size)*cell_size+kp[id,1]%cell_size
            kmap[row_id,col_id,cell_id] = 1
            jud[row_id,col_id] = 0
        kmap = torch.cat([kmap,jud.float().unsqueeze(-1)],dim=-1)
    elif add_gauss:
        kmap = torch.zeros(image_shape,dtype=torch.float32)
        height, width = image_shape[0],image_shape[1]
        for id in range(kp.shape[0]):
            center_x, center_y = kp[id,1],kp[id,0]
            x0 = int(max(0, center_x - rad))
            y0 = int(max(0, center_y - rad))
            x1 = int(min(width, center_x + rad))
            y1 = int(min(height, center_y + rad))
            kmap[y0:y1,x0:x1] = gauss_matrix[rad-(center_y-y0):rad+(y1-center_y),\
                                             rad-(center_x-x0):rad+(x1-center_x)]
            # for y in range(y0, y1):
            #     for x in range(x0, x1):
            #         d = (x - center_x) ** 2 + (y - center_y) ** 2  ###高斯！！
            #         exp = d / 2.0 / sigma / sigma  ##高斯！！
            #         kmap[y][x] = torch.exp(-exp)
    else:
        kmap = torch.zeros(image_shape)
        if kp.shape[0]!=0:
            kmap[kp[:,0],kp[:,1]] = 1
    # kmap = torch.zeros((image_shape))
    # kmap[kp[:,0],kp[:,1]] = 1
    # image_shape = tf.cast(tf.shape(data['image'])[:2], tf.int64)
    # kp = tf.minimum(tf.cast(tf.round(data['keypoints']), tf.int64), image_shape-1)
    # kmap = tf.scatter_nd(kp, tf.ones([tf.shape(kp)[0]], dtype=tf.int64), image_shape)
    return kmap

class MagicDataset(torch.utils.data.Dataset):   
    def __init__(self,config,test_mode = False,on_the_fly = False):
        super(torch.utils.data.Dataset,self).__init__()
        self.scale = True
        self.primitives = [
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
        self.test_mode = test_mode
        self.config = config
        self.data_lenth = self.config['data_lenth']
        self.is_deeptype = self.config['is_deeptype']
        self.add_gauss = self.config['add_gauss']
        self.location = self.config['pic_loc']
        self.test_data_lenth = self.config['test_data_lenth']
        self.ini_gauss()

    def ini_gauss(self):
        # image_shape = self.config['data']['generation']['image_size']
        sigma = 2
        delta = 1
        # height, width = image_shape[0],image_shape[1]
        self.gauss_matrix = torch.zeros((2*delta * sigma+1 , 2*delta * sigma+1))
        self.rad = delta * sigma
        for y in range(2*delta * sigma+1):
            for x in range(2*delta * sigma+1):
                d = (x - delta * sigma) ** 2 + (y - delta * sigma) ** 2  ###高斯！！
                exp = d / 2.0 / sigma / sigma  ##高斯！！
                self.gauss_matrix[y][x]  = torch.exp(torch.tensor(-exp))
    


    def __getitem__(self, item):
        if self.test_mode:
            loc = self.location    
        else:
            with open("dataset_state.txt", 'r') as file:
                lines = file.readlines()  # 读取所有行到一个列表中
                word = lines[-1].split()[0]   # 获取最后一行
                loc = word
        id = item
        pic_name_trans = os.path.join(loc,'trans_{}.png'.format(id))
        pic_name = os.path.join(loc,'ori_{}.png'.format(id))
        homography_path = os.path.join(loc,'homo_{}.npy'.format(id))
        pt_name_trans = os.path.join(loc,'pt_trans_{}.npy'.format(id))
        pt_name = os.path.join(loc,'pt_{}.npy'.format(id))
        image = cv2.imread(pic_name,cv2.IMREAD_COLOR)
        image_trans = cv2.imread(pic_name_trans,cv2.IMREAD_COLOR)
        baseloc = "/home/lizhonghao/ALIKE/magic_point_dataset/pic1/"
        if image is not None:
            image = image.astype(np.float32)
            # H = np.load(homography_path).astype(np.float32)
            H = None
            points = np.load(pt_name).astype(np.float32)
            # warped_points = np.load(pt_name_trans).astype(np.float32)
        else:
            print("unexisted pic ",item)
            image = cv2.imread(baseloc+"ori_2699.png"\
                               ,cv2.IMREAD_COLOR).astype(np.float32)
            H = None
            points = np.load(baseloc+"pt_2699.npy").astype(np.float32)
        if image_trans is not None:
            image_trans = image_trans.astype(np.float32)
            warped_points = np.load(pt_name_trans).astype(np.float32)
        else:
            print("unexisted pic trans ",item)
            image_trans = cv2.imread("/home/lizhonghao/ALIKE/magic_point_dataset/pic1/trans_2699.png"\
                               ,cv2.IMREAD_COLOR).astype(np.float32)
            warped_points = np.load(baseloc+"pt_trans_2699.npy").astype(np.float32)

        if self.config['is_graystyle'] :
            if image.shape[-1] != 1 :
                image = image[:,:,-1:]
                image_trans = image_trans[:,:,-1:]
        else:
            if image.shape[-1] == 1 :
                image = np.repeat(image,3,-1)
                image_trans = np.repeat(image_trans,3,-1)
        h0, w0, _ = image.shape
        h1, w1, _ = image_trans.shape
        keypoint_map0 = add_keypoint_map(image,points,self.gauss_matrix,rad = self.rad,\
                                            add_gauss=self.add_gauss,\
                                            is_deeptype =self.is_deeptype )
        keypoint_map1 = add_keypoint_map(image_trans,warped_points,self.gauss_matrix,rad = self.rad,\
                                            add_gauss=self.add_gauss,\
                                            is_deeptype =self.is_deeptype)
        if self.scale:
            image = image/255.
            image_trans = image_trans/255.

        res = {
        'image0': image.transpose(2, 0, 1),  # [C,H,W]
        'image1': image_trans.transpose(2, 0, 1),  # [C,H,W]
        # 'warp01_params': {'mode': 'homo', 'width': w1, 'height': h1,
        #                     'homography_matrix': np.linalg.inv(H) if H is not None else None},
        # 'warp10_params': {'mode': 'homo', 'width': w0, 'height': h0,
        #                     'homography_matrix': H, },
        'warp01_params': {'mode': 'homo', 'width': w1, 'height': h1,
                            'homography_matrix': np.identity(3),},
        'warp10_params': {'mode': 'homo', 'width': w0, 'height': h0,
                            'homography_matrix': np.identity(3), },
        #"keypoints0": points ,
        #"keypoints1": warped_points,
        "keypoints0_map": keypoint_map0, 
        "keypoints1_map": keypoint_map1, 
        }
        return res

    
        
    def __len__(self):
        if self.test_mode:
            return self.test_data_lenth
        else:
            return self.data_lenth



if __name__ == "__main__":
     magicpoint_dataset = MagicDataset()
     train_loader = torch.utils.data.DataLoader(magicpoint_dataset, batch_size=256, shuffle=False, pin_memory=True,
                              num_workers=1)
     for image,points,valid_mask in train_loader:
          image = image
          points = points
    

    