import glob
import random
from pathlib import Path
from typing import List, Callable, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sacred import Experiment, SETTINGS

from config_training import default_configs, task_name, experiment_id_g

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment(task_name)
ex.config(default_configs)


def decode_fg(width, height, n_channels, n_classes, mask_width=None, mask_height=None):
    if n_channels not in [1, 3]:
        raise ValueError(f"n_channels: {n_channels}")

    if mask_height is None or mask_width is None:
        mask_width, mask_height = width, height

    def parse_function(proto):
        # Create a dictionary describing the features.
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string)
        }
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(proto, features=features)

        image = tf.io.decode_raw(parsed["image"], tf.float32)
        image = tf.reshape(image, (height, width, n_channels))
        mask = tf.io.decode_raw(parsed["mask"], tf.float32)
        mask = tf.reshape(mask, (mask_height, mask_width, n_classes))

        return image, mask

    return parse_function


@ex.capture
def get_image_decoder(input_image_shape: Tuple[int, int, int]) -> Callable:
    return decode_fg(width=input_image_shape[1],
                     height=input_image_shape[0],
                     mask_width=input_image_shape[1],
                     mask_height=input_image_shape[0],
                     n_channels=input_image_shape[2],
                     n_classes=1)


def get_image_mask_dataset(
        filenames: list,
        decoder: Callable,
        batch_size: int,
        buffer_size: int,
):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(
        decoder, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


@ex.capture
def get_train_dataset(
        batch_size: int,
        train_files: dict,
        dataset_path: Path,
        train_buffer_size: int,
) -> tf.data.Dataset:
    tfrecord_files = []
    for k, v in train_files.items():
        for _ in range(v):
            tfrecord_files += list(glob.glob(str(dataset_path.joinpath(k))))
    random.shuffle(tfrecord_files)
    decoder = get_image_decoder()
    return get_image_mask_dataset(tfrecord_files, decoder, batch_size, train_buffer_size)


@ex.automain
def run_main():
    train_ds = get_train_dataset()

    # showing dataset
    for image, mask in train_ds.take(10):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,8))
        f.subplots_adjust(hspace = .2, wspace = .05)

        ax1.imshow(image / 255.)
        ax2.imshow(mask, cmap="gray")
        plt.show()
