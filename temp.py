import torch
from sample_H import sample_homography
import cv2
import numpy as np
import torch.nn.functional as F
def homography_mat(H):
    if len(H.shape)==1: H = H.unsqueeze(0)
    H = torch.concat([H, torch.ones((H.shape[0],1))],dim=1).reshape(-1,3,3)
    return H[0]
def H_transform(image,H,pts1,pts2):
    H = H.numpy().astype(np.float32)
    pts1,pts2 = pts1.numpy().astype(np.int32),pts2.numpy().astype(np.int32)
    H_, status  = cv2.findHomography(pts1, pts2) 
 
    output_image = cv2.warpPerspective(image, H_, (image.shape[1], image.shape[0]))
    #output_image =np.repeat(output_image,3,axis=-1)
    for pt in pts2:
        if (pt>0).all() :
            output_image[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10,:] = np.array([255,0,0])
    for i,pt in enumerate(pts1):
        pt =(H @ np.array([pt[0],pt[1],1]))
        pt =(pt/pt[-1]).astype(np.int32)
        print(pt,pts2[i])
        if (pt>0).all() :
            output_image[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10,:] = np.array([0,255,0])
    for pt in [np.array([480,360])]:
        pt =(H @ np.array([pt[0],pt[1],1]))
        pt =(pt/pt[-1]).astype(np.int32)
        if (pt>0).all() :
            output_image[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10,:] = np.array([0,0,255])
    cv2.imwrite("xsxa.png",output_image)
    x= 12
# def box_nms(prob, size, iou=0.1, threshold=0.01, keep_top_k=0):
#     pts = tf.cast(tf.where(tf.greater_equal(prob, threshold)), dtype=tf.float32)
#     size = tf.constant(size/2.)
#     boxes = tf.concat([pts-size, pts+size], axis=1)
#     scores = tf.gather_nd(prob, tf.cast(pts, dtype=tf.int32))

#     indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)
#     pts = tf.gather(pts, indices)
#     scores = tf.gather(scores, indices)
#     if keep_top_k:
#         k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
#         scores, indices = tf.nn.top_k(scores, k)
#         pts = tf.gather(pts, indices)
#     prob = tf.scatter_nd(tf.cast(pts, tf.int32), scores, tf.shape(prob))

#     return prob


def homography_adaptation(image, model=0, config=0):
    # import time
    # t0 = time.time()
    # probs = model(image)[1]
    # # t1 = time.time()
    # counts = torch.ones(probs.shape)
    # images = image


    # probs = probs.unsqueeze(-1)
    # counts = counts.unsqueeze(-1)
    # images = images.unsqueeze(-1)

    shape = image.shape[0:2]
    H ,pts1 , pts2= sample_homography(shape)
    H_mat = homography_mat(H)
    H_transform(image,H_mat,pts1,pts2)
#    def step(i, probs, counts, images):
        # H = sample_homography(shape, config["model"]
        #                       ['homography_adaptation']['homographies'])
        # H_inv = invert_homography(H)
        # warped = H_transform(image, H, interpolation='BILINEAR')
        # count = H_transform(tf.expand_dims(
        #     tf.ones(tf.shape(image)[:3]), -1), H_inv, interpolation='NEAREST')
        # mask = H_transform(tf.expand_dims(
        #     tf.ones(tf.shape(image)[:3]), -1), H, interpolation='NEAREST')

        # if config["model"]["homography_adaptation"]['valid_border_margin']:
        #     kernel = cv2.getStructuringElement(
        #         cv2.MORPH_ELLIPSE, (config["model"]["homography_adaptation"]['valid_border_margin'] * 2,) * 2)

        #     #消融
        #     count = tf.nn.erosion2d(value=count,
        #                             filters=tf.cast(tf.constant(kernel)
        #                                             [..., tf.newaxis], dtype=tf.float32),
        #                             strides=[1, 1, 1, 1],
        #                             dilations=[1, 1, 1, 1],
        #                             padding='SAME',
        #                             data_format="NHWC")[..., 0] + 1.
        #     mask = tf.nn.erosion2d(value=mask,
        #                            filters=tf.cast(tf.constant(kernel)
        #                                            [..., tf.newaxis], dtype=tf.float32),
        #                            strides=[1, 1, 1, 1],
        #                            dilations=[1, 1, 1, 1],
        #                            padding='SAME',
        #                            data_format="NHWC")[..., 0] + 1.

        # prob = model(warped)[1]
        # prob = prob * mask
        # prob_proj = H_transform(tf.expand_dims(prob, -1), H_inv,
        #                         interpolation='BILINEAR')[..., 0]
        # prob_proj = prob_proj * count

        # probs = tf.concat([probs, tf.expand_dims(prob_proj, -1)], axis=-1)
        # counts = tf.concat([counts, tf.expand_dims(count, -1)], axis=-1)
        # images = tf.concat([images, tf.expand_dims(warped, -1)], axis=-1)
        # return i + 1, probs, counts, images

    # t2 = time.time()
    # _, probs, counts, images = tf.while_loop(
    #     lambda i, p, c, im: tf.less(
    #         i, config["model"]["homography_adaptation"]["num"] - 1),
    #     step,
    #     [0, probs, counts, images],
    #     parallel_iterations=10,
    #     shape_invariants=[
    #         tf.TensorShape([]),
    #         tf.TensorShape([None, None, None, None]),
    #         tf.TensorShape([None, None, None, None]),
    #         tf.TensorShape([None, None, None, 1, None])])
    # # t3 = time.time()

    # counts = tf.reduce_sum(counts, axis=-1)
    # max_prob = tf.reduce_max(probs, axis=-1)
    # mean_prob = tf.reduce_sum(probs, axis=-1) / counts

    # if config["model"]["homography_adaptation"]['aggregation'] == 'max':
    #     prob = max_prob
    # elif config["model"]["homography_adaptation"]['aggregation'] == 'sum':
    #     prob = mean_prob
    # else:
    #     raise ValueError('Unkown aggregation method: {}'.format(
    #         config['model']['homography_adaptation']['aggregation']))

    # if config['model']['homography_adaptation']['filter_counts']:
    #     prob = tf.where(tf.greater_equal(
    #         counts, config['model']['homography_adaptation']['filter_counts']), prob, tf.zeros_like(prob))
    # # print("T01 = {}, T23 = {}".format(t1-t0, t3-t2))
    # return {'prob': prob, 'counts': counts, 'mean_prob': mean_prob, 'input_images': images, 'H_probs': probs}  # debug

# if __name__ == "__main__":
#     while(1):
#         x = homography_adaptation(np.ones((720,960,3 ),dtype=np.float32)*255)
#         print(x)

with torch.no_grad():
    # lis = []
    # for i in range(8):
    #     for j in range(8):
    #         temp = torch.zeros((8,8),dtype=torch.float32)
    #         temp[i,j] = 1
    #         lis.append(temp.unsqueeze(0))
    # res = torch.concat(lis)
    # print(res.shape)
    unfold = torch.nn.Unfold(8,stride=8)
    x = torch.rand((64,1,1024,800))
    # print(unfold(x).shape)
    x = F.cross_entropy(torch.tensor([[1.,1.],[1.,1.]]),torch.tensor([[1.,1.],[1.,1.]]))
    print(x,np.log(2)*6)
    