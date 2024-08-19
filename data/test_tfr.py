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

    return parsed_features, {'Label': labels}

def input_fn(tfr_file_path, num_epochs, shuffle, batch_size, shuffle_buffer_size=10000):
    dataset = tf.data.TFRecordDataset(tfr_file_path)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    # 使用 prefetch 提前加载数据，1 表示缓冲区大小.
    # 尤其是当数据加载和预处理步骤比较耗时时, 可以显著提高训练性能.
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(1)

    return dataset

if __name__ == "__main__":
    # 读取 TFRecord 文件
    tfr_file = './criteo_x4/train.tr.tfrecords'
    dataset = input_fn(tfr_file, num_epochs=1, shuffle=False, batch_size=2)

    # 创建一个迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                example, labels = sess.run(next_element)
                print(labels, example)
                break
        except tf.errors.OutOfRangeError:
            print("End of dataset")
