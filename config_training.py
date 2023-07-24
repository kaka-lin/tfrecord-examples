import uuid
from pathlib import Path

import tensorflow as tf

# some global variables that sacred needs
task_name = "hair-segmentation"
experiment_id_g = task_name[:2] + '-' + str(uuid.uuid4())


def default_configs():
    task = task_name
    experiment_id = experiment_id_g
    tensorflow_version = tf.__version__

    # images, masks config
    dim_width = 256
    dim_height = 256
    mask_width = 256
    mask_height = 256
    n_channels = 3
    n_classes = 1
    input_image_shape = (dim_height, dim_width, n_channels)
    output_mask_shape = (mask_height, mask_width, 1)

    # dataset
    dataset_path = Path(f"data/")
    train_files = {
        # f"figaro1k/train_{dim_width}_{dim_height}_{mask_width}_{mask_height}_{n_channels}*tfrecord": 1,
        f"train_{dim_width}_{dim_height}_{mask_width}_{mask_height}_{n_channels}*tfrecord": 1,
    }
    train_buffer_size = 100

    # training
    epochs = 300
    train_steps = 100
    dev_steps = 50
    validation_steps = 50
    batch_size = 8
