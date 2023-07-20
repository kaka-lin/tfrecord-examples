import uuid
from pathlib import Path

import tensorflow as tf

# some global variables that sacred needs
task_name = "create-tfrecord"
experiment_id_g = task_name[:2] + '-' + str(uuid.uuid4())


def default_configs():
    task = task_name
    experiment_id = experiment_id_g
    tensorflow_version = tf.__version__

    # images, masks config
    width = 256
    height = 256
    mask_width = 256
    mask_height = 256
    n_channels = 3
    n_classes = 1

    # for create tfrecord
    dataset_path = Path(f"data/figaro1k")
    shard_size = 500
