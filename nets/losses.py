import torch
import torch.nn.functional as F
import numpy as np
import time
class PeakyLoss(object):
    """ PeakyLoss to avoid an uniform score map """

    def __init__(self, scores_th: float = 0.1):
        super().__init__()
        self.scores_th = scores_th

    def __call__(self, pred):
        b, c, h, w = pred['scores_map'].shape
        loss_mean = 0
        CNT = 0
        n_original = len(pred['score_dispersity'][0])
        for idx in range(b):
            scores_kpts = pred['scores'][idx][:n_original]
            valid = scores_kpts > self.scores_th
            loss_peaky = pred['score_dispersity'][idx][valid]

            loss_mean = loss_mean + loss_peaky.sum()
            CNT = CNT + len(loss_peaky)

        loss_mean = loss_mean / CNT if CNT != 0 else pred['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class ReprojectionLocLoss(object):
    """
    Reprojection location errors of keypoints to train repeatable detector.
    """

    def __init__(self, norm: int = 1, scores_th: float = 0.1):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        loss_mean = 0
        CNT = 0
        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            if self.norm == 2:
                dist = correspondences[idx]['dist']
            elif self.norm == 1:
                dist = correspondences[idx]['dist_l1']
            else:
                raise TypeError('No such norm in correspondence.')

            ids0_d = correspondences[idx]['ids0_d']
            ids1_d = correspondences[idx]['ids1_d']

            scores0 = correspondences[idx]['scores0'].detach()[ids0_d]
            scores1 = correspondences[idx]['scores1'].detach()[ids1_d]
            valid = (scores0 > self.scores_th) * (scores1 > self.scores_th)
            reprojection_errors = dist[ids0_d, ids1_d][valid]

            loss_mean = loss_mean + reprojection_errors.sum()
            CNT = CNT + len(reprojection_errors)

        loss_mean = loss_mean / CNT if CNT != 0 else correspondences[0]['dist'].new_tensor(0)

        assert not torch.isnan(loss_mean)
        return loss_mean


def local_similarity(descriptor_map, descriptors, kpts_wh, radius):
    """
    :param descriptor_map: CxHxW
    :param descriptors: NxC
    :param kpts_wh: Nx2 (W,H)
    :return:
    """
    _, h, w = descriptor_map.shape
    ksize = 2 * radius + 1

    descriptor_map_unflod = torch.nn.functional.unfold(descriptor_map.unsqueeze(0),
                                                       kernel_size=(ksize, ksize),
                                                       padding=(radius, radius))
    descriptor_map_unflod = descriptor_map_unflod[0].t().reshape(h * w, -1, ksize * ksize)
    # find the correspondence patch
    kpts_wh_long = kpts_wh.detach().long()
    patch_ids = kpts_wh_long[:, 0] + kpts_wh_long[:, 1] * h
    desc_patches = descriptor_map_unflod[patch_ids].permute(0, 2, 1).detach()  # N_kpts x s*s x 128

    local_sim = torch.einsum('nsd,nd->ns', desc_patches, descriptors)
    local_sim_sort = torch.sort(local_sim, dim=1, descending=True).values
    local_sim_sort_mean = local_sim_sort[:, 4:].mean(dim=1)  # 4 is safe radius for bilinear interplation

    return local_sim_sort_mean


class ScoreMapRepLoss(object):
    """ Scoremap repetability"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.radius = 2

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            kpts01 = correspondences[idx]['kpts01']
            kpts10 = correspondences[idx]['kpts10']  # valid warped keypoints

            # =====================
            scores_kpts10 = torch.nn.functional.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kpts01 = torch.nn.functional.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                            mode='bilinear', align_corners=True)[0, 0, 0, :]

            s0 = scores_kpts01 * correspondences[idx]['scores0']  # repeatability
            s1 = scores_kpts10 * correspondences[idx]['scores1']  # repeatability

            # ===================== repetability
            similarity_map_01 = correspondences[idx]['similarity_map_01_valid']
            similarity_map_10 = correspondences[idx]['similarity_map_10_valid']

            pmf01 = ((similarity_map_01.detach() - 1) / self.temperature).exp()
            pmf10 = ((similarity_map_10.detach() - 1) / self.temperature).exp()

            kpts01 = kpts01.detach()
            kpts10 = kpts10.detach()

            pmf01_kpts = torch.nn.functional.grid_sample(pmf01.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts = torch.nn.functional.grid_sample(pmf10.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            repetability01 = torch.diag(pmf01_kpts)
            repetability10 = torch.diag(pmf10_kpts)

            # ===================== reliability
            # ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            # descriptor_map0 = pred0['descriptor_map'][idx].detach()
            # descriptor_map1 = pred1['descriptor_map'][idx].detach()
            # descriptors0 = pred0['descriptors'][idx][ids0].detach()
            # descriptors1 = pred1['descriptors'][idx][ids1].detach()
            # kpts0 = pred0['keypoints'][idx][ids0].detach()
            # kpts1 = pred1['keypoints'][idx][ids1].detach()
            # kpts0_wh = (kpts0 / 2 + 0.5) * wh
            # kpts1_wh = (kpts1 / 2 + 0.5) * wh
            # ls0 = local_similarity(descriptor_map0, descriptors0, kpts0_wh, self.radius)
            # ls1 = local_similarity(descriptor_map1, descriptors1, kpts1_wh, self.radius)
            # reliability0 = 1 - ((ls0 - 1) / self.temperature).exp()
            # reliability1 = 1 - ((ls1 - 1) / self.temperature).exp()

            fs0 = repetability01  # * reliability0
            fs1 = repetability10  # * reliability1

            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                loss_mean = loss_mean + loss01.sum()
                CNT = CNT + len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                loss_mean = loss_mean + loss10.sum()
                CNT = CNT + len(loss10)

        loss_mean = loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class DescReprojectionLoss(object):
    """ Reprojection loss between warp and descriptor matching """

    def __init__(self, temperature=0.02):
        super().__init__()
        self.inv_temp = 1. / temperature

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        device = pred0['scores_map'].device
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            kpts01, kpts10 = correspondences[idx]['kpts01'], correspondences[idx]['kpts10']  # valid warped keypoints

            similarity_map_01 = correspondences[idx]['similarity_map_01']
            similarity_map_10 = correspondences[idx]['similarity_map_10']
            ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            ids0_out, ids1_out = correspondences[idx]['ids0_out'], correspondences[idx]['ids1_out']

            # ======================= valid
            similarity_map_01_valid, similarity_map_10_valid = similarity_map_01[ids0], similarity_map_10[ids1]
            similarity_map_01_valid = (similarity_map_01_valid - 1) * self.inv_temp
            similarity_map_10_valid = (similarity_map_10_valid - 1) * self.inv_temp

            # matching probability mass function
            pmf01_valid = torch.softmax(similarity_map_01_valid.view(-1, h * w), dim=1).view(-1, h, w)
            pmf10_valid = torch.softmax(similarity_map_10_valid.view(-1, h * w), dim=1).view(-1, h, w)

            pmf01_kpts_valid = torch.nn.functional.grid_sample(pmf01_valid.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            pmf10_kpts_valid = torch.nn.functional.grid_sample(pmf10_valid.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                               mode='bilinear', align_corners=True)[0, :, 0, :]
            # as we use the gt correspondence here, the outlier uniform pmf is ignored
            # C_{Q,N} in NRE
            C01 = torch.diag(pmf01_kpts_valid)
            C10 = torch.diag(pmf10_kpts_valid)

            # ======================= out
            similarity_map_01_out, similarity_map_10_out = similarity_map_01[ids0_out], similarity_map_10[ids1_out]
            out0 = torch.ones(len(similarity_map_01_out), device=device)
            out1 = torch.ones(len(similarity_map_10_out), device=device)
            # cat outside scores to similarity_map, thus similarity_map is (N, H*W +1)
            similarity_map_01_out = torch.cat([similarity_map_01_out.reshape(-1, h * w), out0[:, None]], dim=1)
            similarity_map_10_out = torch.cat([similarity_map_10_out.reshape(-1, h * w), out1[:, None]], dim=1)
            similarity_map_01_out = (similarity_map_01_out - 1) * self.inv_temp
            similarity_map_10_out = (similarity_map_10_out - 1) * self.inv_temp
            pmf01_out = torch.softmax(similarity_map_01_out, dim=1)
            pmf10_out = torch.softmax(similarity_map_10_out, dim=1)
            if len(pmf01_out) > 0:
                C01_out = pmf01_out[:, -1]
            else:
                C01_out = C01.new_tensor([])
            if len(pmf10_out) > 0:
                C10_out = pmf10_out[:, -1]
            else:
                C10_out = C10.new_tensor([])

            # ======================= out
            C = torch.cat([C01, C10, C01_out, C10_out])  # C
            C_widetilde = -C.log()  # \widetilde{C}

            loss_mean = loss_mean + C_widetilde.sum()
            CNT = CNT + len(C_widetilde)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_tensor(0)
        assert not torch.isnan(loss_mean)
        return loss_mean


class TripletLoss(object):
    def __init__(self, margin: float = 0.5, neg_mining_pix_th: int = 5):
        super().__init__()
        self.margin = margin
        self.th = neg_mining_pix_th
        self.relu = torch.nn.ReLU()

    def __call__(self, pred0, pred1, correspondences):
        b, c, h, w = pred0['scores_map'].shape
        wh = pred0['keypoints'][0].new_tensor([[w - 1, h - 1]])
        loss_mean = 0
        CNT = 0

        for idx in range(b):
            if correspondences[idx]['correspondence0'] is None:
                continue

            similarity_map_01 = correspondences[idx]['similarity_map_01_valid']
            similarity_map_10 = correspondences[idx]['similarity_map_10_valid']
            kpts01, kpts10 = correspondences[idx]['kpts01'], correspondences[idx]['kpts10']

            # ================= 1. positive
            positive01 = torch.nn.functional.grid_sample(similarity_map_01.unsqueeze(0), kpts01.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            positive10 = torch.nn.functional.grid_sample(similarity_map_10.unsqueeze(0), kpts10.view(1, 1, -1, 2),
                                                         mode='bilinear', align_corners=True)[0, :, 0, :]
            positive0 = torch.diag(positive01)
            positive1 = torch.diag(positive10)

            # ================= 2. mining negatives
            dist = correspondences[idx]['dist']
            ids0, ids1 = correspondences[idx]['ids0'], correspondences[idx]['ids1']
            desc0 = pred0['descriptors'][idx][ids0]
            desc1 = pred1['descriptors'][idx][ids1]

            # a) mining negatives for kpts0 in  kpts1
            invalid = dist < self.th  # safe radius
            cosim01 = desc0 @ desc1.t()
            cosim01[invalid] = -2  # cosine similarity : -1~1, setting invalid to -2 so they will be excluded

            sorted0_values, sorted0_index = cosim01.sort(descending=True)  # N0_valid x N1_valid
            negatives0 = sorted0_values[:, 0]

            sorted1_values, sorted1_index = cosim01.t().sort(descending=True)  # N1_valid x N0_valid
            negatives1 = sorted1_values[:, 0]

            triplet_loss0 = self.relu(self.margin - positive0 + negatives0)
            triplet_loss1 = self.relu(self.margin - positive1 + negatives1)

            loss_mean = loss_mean + triplet_loss0.sum() + triplet_loss1.sum()
            CNT = CNT + len(triplet_loss0) + len(triplet_loss1)

        loss_mean = loss_mean / CNT if CNT != 0 else wh.new_tensor(0)
        return loss_mean



class mgpt_loss(object):
    def __init__(self,cell_size = 8):
        super().__init__()
        self.cell_size = cell_size
        self.cell_counts = self.cell_size*self.cell_size
        self.window = (self.cell_size,self.cell_size)
        self.unfold = torch.nn.Unfold(cell_size,stride=cell_size)
    def __call__(self,logits,keypoint_map,dustbin = None ):
        def change_to_deepth(matrix,is_label=False):
            if len(matrix.shape) != 4:
                matrix = matrix.unsqueeze(1)
                assert len(matrix.shape) == 4
            res =  self.unfold(matrix)   
            if is_label:
                have_kpt = -torch.max(res,dim=-2,keepdim=True).values+1
                res = torch.concat([res,have_kpt],dim=-2)
            return res.permute(0,2,1).reshape(-1,self.cell_counts+is_label)
            # batch_size = matrix.shape[0]
            # row_nums,col_nums= matrix.shape[-2]//8 ,matrix.shape[-1]//8
            # matrix = matrix.reshape(batch_size,-1,8)
            # slices = np.array(range(8))*col_nums
            # kpt_deepth = torch.zeros((batch_size,row_nums,col_nums,\
            #                 self.cell_counts+is_label),dtype=torch.float32,device=matrix.device) 
            # for i in range(row_nums):
            #     for j in range(col_nums):
            #         if is_label:
            #             for k in range(batch_size):
            #                 if (matrix[k,slices+j,:]==0).all():
            #                     kpt_deepth[k,i,j,-1] = 1
            #                 else:
            #                     kpt_deepth[k,i,j,:-1] = matrix[k,slices+j,:].reshape(-1)
            #         else:
            #             kpt_deepth[:,i,j,:] = matrix[:,i*col_nums*8+slices+j,:].reshape(batch_size,-1)
            # return kpt_deepth.reshape(-1,self.cell_counts+is_label) 
        logits = logits[:,self.cell_size:-self.cell_size,self.cell_size:-self.cell_size]
        keypoint_map = keypoint_map[:,self.cell_size:-self.cell_size,self.cell_size:-self.cell_size]
        dustbin = dustbin[:,self.cell_size:-self.cell_size,self.cell_size:-self.cell_size]
        if dustbin != None:
            if keypoint_map.shape[-1] == self.cell_size**2+1:
                keypoint_map = keypoint_map.reshape(-1,self.cell_size**2+1)
            else:
                assert keypoint_map.shape==logits.shape    
                keypoint_map = change_to_deepth(keypoint_map,is_label=True)
            logits = change_to_deepth(logits,is_label=False)
            dustbin = F.avg_pool2d(dustbin, kernel_size=self.window, stride=self.window).reshape(-1,1)
            logits = torch.cat([logits,dustbin],dim = -1)
        else:
            assert keypoint_map.shape==logits.shape
            keypoint_map = change_to_deepth(keypoint_map)
            logits = change_to_deepth(logits)
        weights = self.cal_w(keypoint_map)
        keypoint_map = torch.argmax(keypoint_map,dim=-1)
        loss = F.cross_entropy(logits,keypoint_map,weight=weights)
        return loss
    def cal_w(self,label):
        nums_d = torch.sum(label[:,-1]==1)
        nums_t = label.shape[0]-nums_d
        sum = label.shape[0]
        weights = torch.zeros(label.shape[-1],device=label.device)
        # weights[:] = nums_d/sum
        # weights[-1] = nums_t/sum
        weights[:] = sum/(nums_t+1e-15)
        weights[-1] = sum/(nums_d+1e-15)
        return weights


class mgpt_mseloss(object):
    def __init__(self):
        super().__init__()
        self.crop = True
    def __call__(self,logits,kmap):
        # logits = logits/torch.sum(logits,dim=(-1,-2),keepdim=True)
        assert logits.shape == kmap.shape
        if self.crop:
            logits = logits[:,10:-10,10:-10]
            kmap = kmap[:,10:-10,10:-10]
        # point_counts = torch.sum(logits)/torch.sum(kmap)
        # point_counts = torch.sum(logits**2)/torch.sum(kmap**2)
        logits = logits/torch.sqrt(1e-15+torch.sum(logits**2,dim=(-1,-2),keepdim=True))
        kmap = kmap/torch.sqrt(1e-15+torch.sum(kmap**2,dim=(-1,-2),keepdim=True))
        # logits = logits/torch.sum(logits)
        # kmap = kmap/torch.sum(kmap)
        # pixel_counts = 1
        # for item in logits.shape:
        #     pixel_counts *= item
        return torch.sum(torch.square(logits-kmap))/logits.shape[0]



    
    # logits: 1, 15, 20, 65 --- (64 probs + 1 dustbin)
    # keypoint_map: 1, 120, 160 ---> labels: 1, 15, 20 [H / 8, W / 8]
    # keypoint_map: 1, 120, 160 ---> valid_mask: 1, 15, 20 [H / 8, W / 8]
        
        # row_nums,col_nums= keypoint_map.shape[0] ,keypoint_map.shape[1]
        # loss=0
        # kmparr = []
        # logarr = []
        # # st = time.time()
        # for i in range(row_nums):
        #         for j in range(col_nums):
        #             if (keypoint_map[i,j]==0).all():
        #                 logarr.append(logits[i,j].reshape(-1))
        #                 kmparr.append(keypoint_map[i,j].reshape(-1))
        # en = time.time()
        # print(en-st)
        
        # if len(logits) != 0:
        #     # logits = torch.cat(logarr,dim=0)
        #     # keypoint_map = torch.cat(kmparr,dim=0)
        #     loss += F.cross_entropy(logits,keypoint_map)
        # else:
        #     loss += F.cross_entropy(logits,keypoint_map)

        
    

    