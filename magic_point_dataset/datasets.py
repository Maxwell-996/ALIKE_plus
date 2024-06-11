import tensorflow as tf
import numpy as np
import cv2
import PIL.Image
from dataset.synthetic_shapes import parse_primitives
import dataset.synthetic_data as synthetic_data
from pathlib import Path
from data_utils import read_points, read_image, ratio_preserving_resize, add_dummy_valid_mask, photometric_augmentation, homographic_augmentation, add_keypoint_map, downsample


def build_synthetic_dataset(config, target="training"):
    drawing_primitives = [
        'draw_lines',
        'draw_polygon',
        'draw_multiple_polygons',
        'draw_ellipses',
        'draw_star',
        'draw_checkerboard',
        'draw_stripes',
        'draw_cube',
        'gaussian_noise'
    ]

    def generate_shapes():
        primitives = parse_primitives(config["data"]['primitives'], drawing_primitives)
        while True:
            primitive = np.random.choice(primitives)
            image = synthetic_data.generate_background(
                config["data"]['generation']['image_size'], **config["data"]['generation']['params']['generate_background'])
            points = np.array(getattr(synthetic_data, primitive)(
                image, **config["data"]['generation']['params'].get(primitive, {})))
            yield (np.expand_dims(image, axis=-1).astype(np.float32), np.flip(points.astype(np.float32), 1))

    images = []
    points = []
    if config["data"]["on-the-fly"]:
        dataset = tf.data.Dataset.from_generator(
            generate_shapes,
            (tf.float32, tf.float32),
            (tf.TensorShape(config["data"]["generation"]["image_size"] + [1]), tf.TensorShape([None, 2])))
        dataset = dataset.map(lambda i, c: downsample(
            i, c, **config["data"]["preprocessing"]))
    else:
        dataset_path = config["data"]["dataset"]
        for dp in drawing_primitives:
            images_files = Path(dataset_path, dp, "images", target).glob("*.png")
            points_files = Path(dataset_path, dp, "points", target).glob("*.npy")
            # need to be paired between images and keypoints
            images += sorted([str(file) for file in images_files])
            points += sorted([str(file) for file in points_files])
            dataset = tf.data.Dataset.from_tensor_slices((images, points))
            # shuffle before split
            dataset = dataset.shuffle(len(images), seed=22,
                                      reshuffle_each_iteration=False)
            dataset = dataset.map(lambda image, points: (read_image(
                image, config), tf.numpy_function(read_points, [points], tf.float32)))

            if target == "training":
                dataset = dataset.skip(config["data"]["validation_size"])
                # dataset = dataset.take(config["data"]["validation_size"])
            if target == "validation":
                dataset = dataset.take(config["data"]["validation_size"])

    dataset = dataset.map(lambda image, keypoints: {
                          "image": image, "keypoints": keypoints})
    # segmentation later ?
    dataset = dataset.map(add_dummy_valid_mask)

    if config["data"]["augmentation"]["photometric"]["enable"]:
        dataset = dataset.map(lambda x: photometric_augmentation(x, config))
    if config["data"]["augmentation"]["homographic"]["enable"]:
        dataset = dataset.map(lambda x: homographic_augmentation(
            x, add_homography=False, **config["data"]["augmentation"]["homographic"]))

    dataset = dataset.map(add_keypoint_map)
    dataset = dataset.map(
        lambda d: {**d, "image": tf.cast(d["image"], tf.float32) / 255.})

    dataset = dataset.batch(
        batch_size=config["model"]["batch_size"], drop_remainder=True).prefetch(2)

    return dataset


def build_repeatability_dataset(config):
    dataset_folder = "COCO/patches" if config["data"]["name"] == "coco" else 'HPatches'
    base_path = Path(config["data"]["dataset"], dataset_folder)
    folder_paths = [x for x in base_path.iterdir() if x.is_dir()]
    image_paths = []
    warped_image_paths = []
    homographies = []
    for path in folder_paths:
        if config["data"]['alteration'] == 'i' and path.stem[0] != 'i':
            continue
        if config["data"]['alteration'] == 'v' and path.stem[0] != 'v':
            continue
        num_images = 1 if config["data"]["name"] == "coco" else 5
        file_ext = '.ppm' if config["data"]["name"] == 'hpatches' else '.jpg'
        for i in range(2, 2 + num_images):
            image_paths.append(str(Path(path, "1" + file_ext)))
            warped_image_paths.append(str(Path(path, str(i) + file_ext)))
            homographies.append(np.loadtxt(str(Path(path, "H_1_" + str(i)))))
    if config["data"]['truncate']:
        image_paths = image_paths[:config['truncate']]
        warped_image_paths = warped_image_paths[:config['truncate']]
        homographies = homographies[:config['truncate']]
    files = {'image_paths': image_paths,
             'warped_image_paths': warped_image_paths,
             'homography': homographies}

    def _adapt_homography_to_preprocessing(zip_data):
        H = tf.cast(zip_data['homography'], tf.float32)
        source_size = tf.cast(zip_data['shape'], tf.float32)
        source_warped_size = tf.cast(zip_data['warped_shape'], tf.float32)
        target_size = tf.cast(tf.convert_to_tensor(config["data"]['preprocessing']['resize']),
                              tf.float32)

        # Compute the scaling ratio due to the resizing for both images
        s = tf.reduce_max(tf.divide(target_size, source_size))
        up_scale = tf.linalg.diag(tf.stack([1. / s, 1. / s, tf.constant(1.)]))
        warped_s = tf.reduce_max(tf.divide(target_size, source_warped_size))
        down_scale = tf.linalg.diag(tf.stack([warped_s, warped_s, tf.constant(1.)]))

        # Compute the translation due to the crop for both images
        pad_y = tf.cast(
            ((source_size[0] * s - target_size[0]) / tf.constant(2.0)), tf.int32)
        pad_x = tf.cast(
            ((source_size[1] * s - target_size[1]) / tf.constant(2.0)), tf.int32)
        translation = tf.stack([tf.constant(1), tf.constant(0), pad_x,
                                tf.constant(0), tf.constant(1), pad_y,
                                tf.constant(0), tf.constant(0), tf.constant(1)])
        translation = tf.cast(tf.reshape(translation, [3, 3]), tf.float32)
        pad_y = tf.cast(((source_warped_size[0] * warped_s - target_size[0])
                         / tf.constant(2.0)), tf.int32)
        pad_x = tf.cast(((source_warped_size[1] * warped_s - target_size[1])
                         / tf.constant(2.0)), tf.int32)
        warped_translation = tf.stack([tf.constant(1), tf.constant(0), -pad_x,
                                       tf.constant(0), tf.constant(1), -pad_y,
                                       tf.constant(0), tf.constant(0), tf.constant(1)])
        warped_translation = tf.cast(tf.reshape(warped_translation, [3, 3]), tf.float32)

        H = warped_translation @ down_scale @ H @ up_scale @ translation
        return H

    shapes, images, names = [], [], []
    for image_path in files["image_paths"]:
        image = cv2.imread(image_path)
        image = tf.image.rgb_to_grayscale(image)
        if config["data"]['preprocessing']['resize']:
            image = ratio_preserving_resize(
                image, config["data"]['preprocessing']["resize"])
        shapes.append(image.shape[:2])
        images.append(image)
        names.append("_".join(image_path.split("/")[-2:]))
    shapes = tf.data.Dataset.from_tensor_slices(shapes)
    images = tf.data.Dataset.from_tensor_slices(images)
    names = tf.data.Dataset.from_tensor_slices(names)
    images = images.map(lambda img: tf.cast(img, tf.float32) / 255.)

    warped_shapes, warped_images = [], []
    for warped_image_path in files["warped_image_paths"]:
        warped_image = cv2.imread(warped_image_path)
        warped_image = tf.image.rgb_to_grayscale(warped_image)
        if config["data"]['preprocessing']['resize']:
            warped_image = ratio_preserving_resize(
                warped_image, config["data"]['preprocessing']["resize"])
        warped_shapes.append(warped_image.shape[:2])
        warped_images.append(warped_image)
    warped_shapes = tf.data.Dataset.from_tensor_slices(warped_shapes)
    warped_images = tf.data.Dataset.from_tensor_slices(warped_images)
    warped_images = warped_images.map(lambda img: tf.cast(img, tf.float32) / 255.)

    homographies = tf.data.Dataset.from_tensor_slices(np.array(files['homography']))
    if config["data"]['preprocessing']['resize']:
        homographies = tf.data.Dataset.zip({'homography': homographies,
                                            'shape': shapes,
                                            'warped_shape': warped_shapes})
        homographies = homographies.map(_adapt_homography_to_preprocessing)

    dataset = tf.data.Dataset.zip({'image': images, 'warped_image': warped_images,
                                   'homography': homographies, "name": names})

    dataset = dataset.batch(
        batch_size=config["model"]["batch_size"]).prefetch(2)

    return dataset


def build_coco_dataset(config):
    print("coco path = {}, pseudo labels = {}".format(
        config["data"]["path"], config["data"]["labels"]))
    coco_path = Path(config["data"]["path"], "train2017")
    image_paths = list(coco_path.iterdir())
    image_paths = [str(p) for p in image_paths if p.suffix == ".jpg"]
    names = [p.split("/")[-1].strip(".jpg") for p in image_paths]
    label_paths = []
    for idx, n in enumerate(names):
        p = Path(config["data"]['labels'], '{}.npz'.format(n))
        if not p.exists():
            # print('Image {} has no corresponding label {}'.format(n, p))
            image_paths.pop(idx)
            names.pop(idx)
            continue
        label_paths.append(str(p))
    files = {'image_paths': image_paths, 'names': names, "label_paths": label_paths}

    images = tf.data.Dataset.from_tensor_slices(files['image_paths'])
    images = images.map(lambda image_path: read_image(
        image_path, config, image_channels=3))
    names = tf.data.Dataset.from_tensor_slices(files['names'])
    points = tf.data.Dataset.from_tensor_slices(files['label_paths'])
    points = points.map(lambda points: tf.numpy_function(
        read_points, [points], tf.float32))
    dataset = tf.data.Dataset.zip({"image": images, "name": names, "keypoints": points})
    dataset = dataset.ignore_errors()

    dataset = dataset.map(add_dummy_valid_mask)

    if config["data"]['cache_in_memory']:
        tf.logging.info('Caching data, first access will take some time.')
        dataset = dataset.cache()

    # make sure to call shuffle after calling cache.
    # dataset = dataset.shuffle(len(images), reshuffle_each_iteration=False)
    train_dataset = dataset.skip(config["data"]['validation_size'])
    val_dataset = dataset.take(config["data"]['validation_size'])

    # Generate the warped pair and apply augmentation
    if config["data"]['warped_pair']['enable']:
        # use num_parallel?
        train_warped = train_dataset.map(lambda d: homographic_augmentation(
            d, add_homography=True, **config["data"]['warped_pair']))
        val_warped = val_dataset.map(lambda d: homographic_augmentation(
            d, add_homography=True, **config["data"]['warped_pair']))
        if config["data"]['augmentation']['photometric']['enable']:
            train_warped = train_warped.map(
                lambda d: photometric_augmentation(d, config))
        train_warped = train_warped.map(add_keypoint_map)
        train_dataset = tf.data.Dataset.zip((train_dataset, train_warped))
        train_dataset = train_dataset.map(lambda d, w: {**d, 'warped': w})
        val_warped = val_warped.map(add_keypoint_map)
        # Merge with the original data
        val_dataset = tf.data.Dataset.zip((val_dataset, val_warped))
        val_dataset = val_dataset.map(lambda d, w: {**d, 'warped': w})

    # Data augmentation
    if config["data"]['augmentation']['photometric']['enable']:
        train_dataset = train_dataset.map(lambda d: photometric_augmentation(d, config))
    if config["data"]['augmentation']['homographic']['enable']:
        train_dataset = train_dataset.map(lambda d: homographic_augmentation(
            d, add_homography=False, **config["data"]['augmentation']['homographic']))

    def post_process(dataset, config):
        dataset = dataset.map(add_keypoint_map)
        dataset = dataset.map(
            lambda d: {**d, 'image': tf.cast(d['image'], tf.float32) / 255.})
        if config["data"]['warped_pair']['enable']:
            dataset = dataset.map(lambda d: {
                                  **d, 'warped': {**d['warped'], 'image': tf.cast(d['warped']['image'], tf.float32) / 255.}})
        dataset = dataset.batch(
            batch_size=config["model"]["batch_size"], drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # Generate the keypoint map
    train_dataset = post_process(train_dataset, config)
    val_dataset = post_process(val_dataset, config)

    return train_dataset, val_dataset
