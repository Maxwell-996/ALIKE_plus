import cv2
import numpy as np
import magic_point_dataset.photometric_augmentation as photoaug
from magic_point_dataset.homograhic_augmentation import sample_homography, compute_valid_mask, warp_points, filter_points
import tensorflow_addons as tfa
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
def photometric_augmentation(img, config):
    primitives = config["data"]["augmentation"]["photometric"]["primitives"]
    params = config["data"]["augmentation"]["photometric"]["params"]

    prim_configs = [params.get(p, {}) for p in primitives]
    for j, (p, c) in enumerate(zip(primitives, prim_configs)):
        img = getattr(photoaug, p)(img, **c)

    return img


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




