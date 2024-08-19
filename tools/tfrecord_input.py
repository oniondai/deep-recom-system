# coding: utf-8
import tensorflow as tf

def _example_parser(tfr_data):
    # 定义Feature结构，需与存储TFRecord时一致
    sparse_feature_names = ['C' + str(i) for i in range(1, 27)]
    dense_feature_names = ['I' + str(i) for i in range(1, 14)]
    label_name = ['Label']

    # 定义使用 tf.parse_single_example 解析的特征字典
    feature_description = {}

    # 定义类别sparse特征的解析方式
    for category_feature in sparse_feature_names:
        feature_description[category_feature] = tf.FixedLenFeature(dtype=tf.string, shape=[1])

    # 定义数值dense特征的解析方式
    for dense_feature in dense_feature_names:
        feature_description[dense_feature] = tf.FixedLenFeature(dtype=tf.float32, shape=[1])

    # 定义标签特征的解析方式
    for label in label_name:
        feature_description[label] = tf.FixedLenFeature(dtype=tf.float32, shape=[1])

    # 解析样本
    parsed_features = tf.parse_single_example(tfr_data, feature_description)

    labels = parsed_features.pop('Label')

    return parsed_features, {"Label": labels}

def input_fn_tfr(tfr_file_path, num_epochs, shuffle, batch_size, shuffle_buffer_size=10000):
    dataset = tf.data.TFRecordDataset(tfr_file_path)
    dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # 解耦数据读取和模型训练

    return dataset
