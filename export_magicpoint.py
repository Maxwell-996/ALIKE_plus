import os
import sys
import cv2
import yaml
import numpy as np
import tensorflow as tf
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from training.LET_wrapper import LET_Wrapper 
from data_utils import add_dummy_valid_mask, add_keypoint_map, homography_adaptation, ratio_preserving_resize, photometric_augmentation, homographic_augmentation, box_nms

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
# if len(gpus) > 1:
#     try:
#         print("Activate Multi GPU")
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#     except RuntimeError as e:
#         print(e)

# else:
try:
    print("Activate Sigle GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
except RuntimeError as e:
    print(e)


def build_dataset(data_dir, config, gpuid=0):
    base_path = Path(data_dir, "train2017/")
    image_paths = list(base_path.iterdir())
    if config["data"]["truncate"]:
        image_paths = image_paths[:config["data"]["truncate"]]
    names = sorted([p.stem for p in image_paths])
    image_paths = sorted([str(p) for p in image_paths])

    # seperate data on multi gpus
    split_length = len(image_paths) // 4  # 4GPUs change
    image_paths = image_paths[gpuid * split_length: (gpuid+1) * split_length]
    names = names[gpuid * split_length: (gpuid+1) * split_length]

    files = {"image_paths": image_paths, "names": names}
    return files


def read_points(filename):
    return np.load(filename.decode("utf-8"))["points"].astype(np.float32)


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    return tf.cast(image, tf.float32)


def preprocess(image):
    image = tf.image.rgb_to_grayscale(image)
    if config["data"]["preprocessing"]:
        image = ratio_preserving_resize(image, config["data"]["preprocessing"]["resize"])

    return image


def make_tf_dataset(files, is_train):
    has_keypoints = "label_paths" in files

    names = tf.data.Dataset.from_tensor_slices(files["names"])
    images = tf.data.Dataset.from_tensor_slices(files["image_paths"])
    images = images.map(read_image)
    images = images.map(preprocess)
    data = tf.data.Dataset.zip({"image": images, "name": names})

    if has_keypoints:
        keypoint = tf.data.Dataset.from_tensor_slices(files["label_paths"])
        keypoint = keypoint.map(lambda path: tf.numpy_function(
            read_points, [path], tf.float32))
        kepoint = keypoint.map(lambda points: tf.reshape(points, [-1, 2]))
        data = tf.data.Dataset.zip((data, kepoint).map(
            lambda d, k: {**d, "keypoints": k}))
        data = data.map(add_dummy_valid_mask)

    if config["data"]["warped_pair"]["enable"]:
        assert has_keypoints
        warped = data.map(lambda d: homographic_augmentation(
            d, config["data"]["warped_pair"], add_homography=True))

        if is_train and config["data"]["augmentatioin"]["photometric"]["enable"]:
            warped = warped.map(
                lambda d: d, config["data"]["augmentation"]["photometric"])

        warped = warped.map(add_keypoint_map)
        data = tf.data.Dataset.zip((data, warped))
        data = data.map(lambda d, w: {**d, "warped": w})

    if has_keypoints and is_train:
        if config["data"]["augmentation"]["photometric"]["enable"]:
            data = data.map(lambda d: photometric_augmentation(
                d, config["data"]["augmentation"]["photometric"]))
        if config["data"]["augmentation"]["homographic"]["enable"]:
            assert not config["data"]["warped_pair"]["enable"]
            data = data.map(lambda d: homographic_augmentation(
                d, config["data"]["augmentation"]["homographic"]))

    if has_keypoints:
        data = data.map(add_dummy_valid_mask)

    data = data.map(lambda d: {**d, "image": tf.cast(d["image"], tf.float32) / 255.})

    if config["data"]["warped_pair"]["enable"]:
        data = data.map(lambda d: {
                        **d, "warped": {**d["warped"], "image": tf.cast(d["warped"]["image"], tf.float32) / 255.}})

    return data


def draw_keypoints(img, corners, color):
    keypoints = [cv2.KeyPoint(int(c[1]), int(c[0]), 1) for c in np.stack(corners)]
    return cv2.drawKeypoints(img.astype(np.uint8), keypoints, None, color=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('gpuid', type=int)
    pretrained_model = 'nopt.ckpt'
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    files = build_dataset(config["path"]["coco_path"], config, args.gpuid)
    model = LET_Wrapper(pretrained_model = pretrained_model )
    dataset = make_tf_dataset(files, False)
    iterator = iter(dataset)

    if not os.path.isdir(config["path"]["output_path"] + "/samples"):
        os.makedirs(config["path"]["output_path"] + "/samples")

    i = 0
    steps = len(files["image_paths"])
    pbar = tqdm(total=steps)
    while True:
        data = []
        image, name = None, None
        try:
            for _ in range(config["model"]["batch_size"]):
                current_data = iterator.get_next()
                data.append(current_data)
                # image = (current_data["image"].numpy()[..., 0] * 255)
                # name = current_data["name"].numpy().decode("utf-8")
        except (StopIteration, tf.errors.OutOfRangeError):
            if not data:
                break
            data += [data[-1] for _ in range(config["model"]["batch_size"] - len(data))]

        data = dict(zip(data[0], zip(*[d.values() for d in data])))
        batch_tensor = tf.stack(data["image"], axis=0)
        prediction = homography_adaptation(batch_tensor, model, config)
        prob = tf.map_fn(lambda p: box_nms(
            p, config["model"]["nms_size"]), prediction["prob"])
        prob = tf.cast(tf.greater_equal(
            prob, config["model"]["threshold"]), dtype=tf.int32)
        pred = {'points': [np.array(np.where(e)).T for e in prob]}

        if len(pred["points"][0]) > 0:
            # result = draw_keypoints(image, pred["points"][0], (0, 255,0))
            # cv2.imwrite(config["path"]["output_path"] + "/samples" + f"/{name}_result.jpg", result)

            def d2l(d): return [dict(zip(d, e)) for e in zip(*d.values())]
            # data: len(data["image"]) == 32
            for p, d in zip(d2l(pred), d2l(data)):
                # import ipdb; ipdb.set_trace()
                if not ('name' in d):
                    p.update(d)
                filename = d['name'].numpy().decode('utf-8') if 'name' in d else str(i)
                filepath = Path(config["path"]["output_path"], '{}.npz'.format(filename))
                np.savez_compressed(filepath, **p)

                # img = data["image"][0].numpy() # 240 320 1
                # img = img * 255
                # img = img.astype(np.int8)
                # points = pred["points"][0] # n * 2
                # imgpath = os.path.join("/home/share/datasets/features/COCO/train2017", filename + ".jpg")
                # img = cv2.imread(imgpath)
                # img = cv2.resize(img, (320, 240))
                # for i in range(p["points"].shape[0]):
                #     h = p["points"][i][0]
                #     w = p["points"][i][1]
                #     cv2.circle(img, (int(w), int(h)), 1, (0, 0, 255))
                # cv2.imwrite(str(filepath).replace(".npz", ".png"), img)

                i += 1
                pbar.update(1)

            if i >= steps:
                print("DONE")
                break
