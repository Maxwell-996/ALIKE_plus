import glob
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os,yaml
import torch.utils.data as data
wdic_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
para_loc =os.path.join(wdic_path, "magic_point_dataset/config.yaml")
with open(para_loc, "r") as f:
    config = yaml.safe_load(f)

def add_keypoint_map(img, points ,gauss_matrix,rad = 3,is_deeptype=False, add_gauss =False ,cell_size = 8):
    assert (is_deeptype and add_gauss) != True
    img = torch.tensor(img)
    image_shape = img.shape[:2]
    if not isinstance(points,torch.Tensor):
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
    else:
        kmap = torch.zeros(image_shape)
        if kp.shape[0]!=0:
            kmap[kp[:,0],kp[:,1]] = 1
    return kmap

class HPatchesDataset(data.Dataset):
    def __init__(self, root: str = '../data/hpatches', alteration: str = 'all' , mode = "test"):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root
        self.mode = mode
        # get all image file name
        self.image0_list = []
        self.image1_list = []
        self.homographies = []
        self.kpts0 = []
        self.kpts1 = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue
            # count images
            file_ext = '.ppm'
            pattern = folder / ('*' + file_ext)
            img_names = glob.glob(pathname=str(pattern))
            num_images = len(img_names)
            # get image pair file names and homographies
            #
            for i in range(2, 1 + num_images):
                self.kpts0.append(str(Path(folder, 'kpts1_' + str(i)+'.pt')))
                self.image0_list.append(str(Path(folder, '1' + file_ext)))
                self.image1_list.append(str(Path(folder, str(i) + file_ext)))
                self.homographies.append(str(Path(folder, 'H_1_' + str(i))))
                self.kpts1.append(str(Path(folder, f'kpts{i}_' + str(i)+'.pt')))
        self.len = len(self.image0_list)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'
        self.config = config
        self.data_lenth = self.config['data_lenth']
        self.is_deeptype = self.config['is_deeptype']
        self.add_gauss = self.config['add_gauss']
        # self.location = self.config['pic_loc']
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
        # read image
        #item = item*5
        img0 = cv2.imread(self.image0_list[item], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.image1_list[item])
        assert img0 is not None, 'can not load: ' + self.image0_list[item]
        assert img1 is not None, 'can not load: ' + self.image1_list[item]

        # bgr -> rgb
        base = 255.
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype('float32') / base  # HxWxC
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype('float32') / base  # HxWxC
        
        img0 = cv2.resize(img0, tuple(config['data']['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, tuple(config['data']['preprocessing']['resize'][::-1]), interpolation=cv2.INTER_LINEAR)
        h0, w0, _ = img0.shape
        h1, w1, _ = img1.shape

        # read homography
        homography = np.loadtxt(self.homographies[item]).astype('float32')
        if True:
            kpts0 = self.kpts0[item]
            kpts1 = self.kpts1[item]
        if self.mode == "test":
        # pack return dict
            return {'image0': img0.transpose(2, 0, 1),  # [C,H,W]
                    'image1': img1.transpose(2, 0, 1),  # [C,H,W]
                    'warp01_params': {'mode': 'homo', 'width': w1, 'height': h1,
                                    'homography_matrix': homography, },
                    'warp10_params': {'mode': 'homo', 'width': w0, 'height': h0,
                                    'homography_matrix': np.linalg.inv(homography), },
                    "kpts0" :kpts0 ,  
                    "kpts1" :kpts1 ,                  
                    }
        elif self.mode == "train":
            kpts0 = torch.load(self.kpts0[item],map_location = "cpu")
            kpts1 = torch.load(self.kpts1[item],map_location = "cpu")
            keypoint_map0 = add_keypoint_map(img0,kpts0,self.gauss_matrix,rad = self.rad,\
                                            add_gauss=self.add_gauss,\
                                            is_deeptype =self.is_deeptype )
            keypoint_map1 = add_keypoint_map(img1,kpts1,self.gauss_matrix,rad = self.rad,\
                                            add_gauss=self.add_gauss,\
                                            is_deeptype =self.is_deeptype)
        # pack return dict
            return {'image0': img0.transpose(2, 0, 1),  # [C,H,W]
                    'image1': img1.transpose(2, 0, 1),  # [C,H,W]
                    'warp01_params': {'mode': 'homo', 'width': w1, 'height': h1,
                                    'homography_matrix': homography, },
                    'warp10_params': {'mode': 'homo', 'width': w0, 'height': h0,
                                    'homography_matrix': np.linalg.inv(homography), },
                    "keypoints0_map": keypoint_map0, 
                    "keypoints1_map": keypoint_map1,                  
                    }

    def __len__(self):
        if self.mode == "test":
            return self.len
        else:
            return self.len
        

    def name(self):
        return self.__class__


class HPatchesSquenceDataset(data.Dataset):
    def __init__(self, root: str = '../data/hpatches', alteration: str = 'all',
                 gray: bool = False):
        """
        Args:
            root: dataset root path
            alteration: # 'all', 'i' for illumination or 'v' for viewpoint
        """
        assert (Path(root).exists()), f"Dataset root path {root} dose not exist!"
        self.root = root
        self.gray = gray

        # get all image file name
        self.image_list = []
        folders = [x for x in Path(self.root).iterdir() if x.is_dir()]
        for folder in folders:
            if alteration == 'i' and folder.stem[0] != 'i':
                continue
            if alteration == 'v' and folder.stem[0] != 'v':
                continue
            # count images
            file_ext = '.ppm'
            pattern = folder / ('*' + file_ext)
            img_names = glob.glob(pathname=str(pattern))
            self.image_list += img_names

        self.len = len(self.image_list)
        assert (self.len > 0), f'Can not find PatchDataset in path {self.root}'

    def __getitem__(self, item):
        # read image
        image0 = cv2.imread(self.image_list[item], cv2.IMREAD_COLOR)
        assert image0 is not None, 'can not load: ' + self.image_list[item]

        # bgr -> rgb
        image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB).astype('float32') / 255.  # HxWxC

        # pack return dict
        ret = {'image': image.transpose(2, 0, 1),  # [C,H,W]
               'path': self.image_list[item]}

        if self.gray:
            gray = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY).astype('float32') / 255.  # HxW
            ret['gray'] = gray[np.newaxis, ...]  # [1,H,W]

        return ret

    def __len__(self):
        return self.len

    def name(self):
        return self.__class__


if __name__ == '__main__':
    from tqdm import tqdm

    hpatches_dataset = HPatchesSquenceDataset(root='../data/hpatches', alteration='i')
    max_shapes = []
    for data in tqdm(hpatches_dataset):
        plt.imshow(data['image'].transpose(1, 2, 0))
        plt.show()
