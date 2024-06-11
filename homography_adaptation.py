import torch
from sample_H import sample_homography
import cv2
import numpy as np
def homography_mat(H):
    if len(H.shape)==1: H = H.unsqueeze(0)
    H = torch.concat([H, torch.ones((H.shape[0],1))],dim=1).reshape(-1,3,3)
    return H[0]

def img_Htransform(image,H,flags = cv2.INTER_NEAREST):
    device = image.device
    if len(image.shape) != 3:
        image = image.squeeze(0)
        assert len(image.shape) == 3
    H = H.cpu().numpy().astype(np.float32)
    image = image.cpu().numpy().astype(np.float32)
    output_image = cv2.warpPerspective(image.transpose(1,2,0), H, (image.shape[-1], image.shape[-2]),flags = flags)
    if len(output_image.shape) != 3:
        output_image = output_image[... ,np.newaxis]
        assert len(output_image.shape) == 3
    return torch.tensor(output_image.transpose(2,0,1),dtype=torch.float32,device=device).unsqueeze(0)

def pt_Htransform(pt,H):
    H = H.numpy().astype(np.float32)

def homography_adaptation(image, model, config):
    # import time
    # t0 = time.time()
    device = image.device
    probs = model(image)['scores_map']
    # t1 = time.time()
    counts = torch.ones(image.shape[-2:],device = device).unsqueeze(0)
    images = image.clone()

    #probs = probs.unsqueeze(0)
    counts = counts.unsqueeze(0)
    #images = images.unsqueeze(0)

    shape = image.shape[-2:]

    def step(i, probs, counts, images):
        H  , _ , __ = sample_homography(shape, config["model"]
                              ['homography_adaptation']['homographies'])
        H_mat= homography_mat(H)
        H_inv = torch.linalg.inv(H_mat)
        warped = img_Htransform(image, H_mat,flags=cv2.INTER_LINEAR)
        count = img_Htransform(torch.ones(image.shape[-2:],device = device).unsqueeze(0), H_inv)
        mask = img_Htransform(torch.ones(image.shape[-2:],device = device).unsqueeze(0), H_mat)

        prob = model(warped)['scores_map']
        prob = prob * mask
        prob_proj = img_Htransform(prob, H_inv,flags=cv2.INTER_LINEAR)
        prob_proj = prob_proj * count


        probs = torch.concat([probs, prob_proj], axis=0)
        counts = torch.concat([counts, count], axis=0)
        images = torch.concat([images, warped], axis=0)
        return i + 1, probs, counts, images

    # t2 = time.time()
    i = 0
    while i < config["model"]["homography_adaptation"]["num"] - 1:
        i, probs, counts, images = step(i, probs, counts, images)


    counts = torch.sum(counts,dim = 0)
    max_prob = torch.max(probs, dim = 0)
    mean_prob = torch.mean(probs, dim = 0) / counts

    if config["model"]["homography_adaptation"]['aggregation'] == 'max':
        prob = max_prob
    elif config["model"]["homography_adaptation"]['aggregation'] == 'sum':
        prob = mean_prob
    else:
        raise ValueError('Unkown aggregation method: {}'.format(
            config['model']['homography_adaptation']['aggregation']))

    if config['model']['homography_adaptation']['filter_counts']:
        prob[prob<config['model']['homography_adaptation']['filter_counts']] = 0
    # print("T01 = {}, T23 = {}".format(t1-t0, t3-t2))
    if len(prob.shape) != 4:
        prob = prob.unsqueeze(0)
        assert len(prob.shape) == 4
    return {'prob': prob, 'counts': counts, 'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug

if __name__ == "__main__":
    while(1):
        x = homography_adaptation(np.ones((720,960,3),dtype=np.float32)*255)
        print(x)