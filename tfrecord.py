import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sacred import Experiment, SETTINGS
import matplotlib.pyplot as plt

from utils.datasets import guided_filter
from config_tfrecord import default_configs

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment('convert-to-tfrecord')
ex.config(default_configs)


def chunks(l, n=1):
    n = max(1, n)
    return list(l[i:i + n] for i in range(0, len(l), n))


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0].split("-")[0]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image_fg(width, height, n_channels, return_shape_ori=False):
    if n_channels not in [1, 3]:
        raise ValueError(f"n_channels: {n_channels}")

    def load_image(image_path, shape=(width, height)):
        image = cv2.imread(image_path)
        if n_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            if image.ndim < 2:
                raise ValueError(image)
            elif image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image[..., :3]
        shape_ori = image.shape
        image = cv2.resize(image, shape)
        image = image.astype(np.float32)
        if return_shape_ori:
            return image, shape_ori
        return image

    return load_image


def load_mask_fg(width, height, dtype=np.int32, rad=None, normalized=True, soft=True):
    def load_mask(mask_path, img, shape=(width, height)):
        mat_format = mask_path.lower().endswith('.mat')

        if mat_format:
            mask = loadmat(mask_path)['mask']
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mask = mask.astype(np.float32) * 255
        else:
            mask = cv2.imread(mask_path)

        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        elif mask.ndim == 3:
            mask = np.expand_dims(mask[..., 0], axis=-1)
        else:
            raise ValueError(f"mask ndim: {mask.ndim}")

        if soft:
            mask = guided_filter(img, mask)

        if normalized:
            mask = mask / 255.

        allow_types = [np.int32, np.float32, bool]
        if dtype not in allow_types:
            raise ValueError(f"only support mask types: {allow_types}")

        if dtype in [np.int32, bool]:
            mask = np.round(mask)

        return mask.astype(dtype)

    return load_mask


def create_tf_example(image, mask):
    image_raw = image.tobytes()
    mask_raw = mask.tobytes()

    feature = {
        "image": _bytes_feature(image_raw),
        "mask": _bytes_feature(mask_raw),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


@ex.capture
def convert_to_tfrecord(dataset_path, width, height, n_channels, mask_width=None, mask_height=None,
                        shard_size=500, datasets_choose="figaro", split_name="train"):
    if mask_width is None or mask_height is None:
        mask_width, mask_height = width, height

    # load and shuffle image/mask pairs
    dataset_path = Path(dataset_path)
    image_paths = sorted(glob.glob(str(dataset_path.joinpath(f"images/{split_name}/*"))))
    mask_paths = sorted(glob.glob(str(dataset_path.joinpath(f"masks/{split_name}/*"))))
    all_pairs = list(zip(image_paths, mask_paths))
    random.shuffle(all_pairs)
    shard_pairs = chunks(all_pairs, shard_size)

    # image/mask preprocessing
    load_image = load_image_fg(width, height, n_channels)
    load_mask = load_mask_fg(mask_width, mask_height)

    for shard_id, pairs in enumerate(shard_pairs):
        image_paths, mask_paths = list(zip(*pairs))

        # setting for tfrecord
        dataset_name = f"{split_name}_{width}_{height}_{mask_width}_{mask_height}_{n_channels}.{shard_id}"
        tfrecord_path = str(dataset_path.joinpath(f"{dataset_name}.tfrecord"))
       #stats_path = str(dataset_path.joinpath(f"{dataset_name}.json"))

        # TFRecord writer
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for i, image_path in enumerate(image_paths):
                mask_path = mask_paths[i]
                if get_basename(image_path) not in get_basename(mask_path):
                    raise ValueError(f"inconsistent basename: \nimage: {image_path}\nmask: {mask_path}")

                try:
                    # image/mask preprocessing
                    image = load_image(image_path)
                    mask = load_mask(mask_path, image)
                except Exception as e:
                    print(str(e))
                    continue

                # convert image/mask to `tf.Example`
                example = create_tf_example(image, mask)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
        print(f"tfrecord: {tfrecord_path} is created.")


@ex.automain
def run_convert():
    convert_to_tfrecord()
