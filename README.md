# Example of tfrecord

An example of TFRecord and using [Sacred](https://github.com/IDSIA/sacred) for configuring our experiments setting.

## Prepare the Dataset (tfrecord)
#### 1. Downloading dataset

```bash
$ sh data/figaro.sh
```

#### 2. Convert the dataset to tfrecord

```bash
$ python3 tfrecord.py
```
