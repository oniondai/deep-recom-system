# coding: utf-8
import tensorflow as tf
from tensorflow import feature_column as fc

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

    return parsed_features, labels

def input_fn_tfr(tfr_file_path, num_epochs, shuffle, batch_size, shuffle_buffer_size=10000):
    dataset = tf.data.TFRecordDataset(tfr_file_path)
    dataset = dataset.map(_example_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # 解耦数据读取和模型训练

    return dataset

# 特征定义
def build_feature_columns():
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    # 对于sparse类别特征，转成embed.
    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(fc.embedding_column(
            fc.categorical_column_with_hash_bucket(feat, 1000), 4))
        #linear_feature_columns.append(fc.categorical_column_with_hash_bucket(feat, 1000))

    # 对于dense特征，使用原值.
    for feat in dense_features:
        #dnn_feature_columns.append(fc.numeric_column(feat))
        linear_feature_columns.append(fc.numeric_column(feat))

    return dnn_feature_columns, linear_feature_columns


if __name__ == "__main__":
    # 读取 TFRecord 文件
    tfr_file = './criteo_x4/train.tr.tfrecords'
    dataset = input_fn_tfr(tfr_file, num_epochs=1, shuffle=False, batch_size=2)

    # 创建一个迭代器
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                example, labels = sess.run(next_element)
                break
        except tf.errors.OutOfRangeError:
            print("End of dataset")

    dnn_feature_columns, linear_feature_columns = build_feature_columns()
    features = example
    deep_input = fc.input_layer(features, dnn_feature_columns)
    # batch size=2，特征一共26个，每个特征4个维度，26*4=104
    print(deep_input) # Tensor("input_layer_3/concat:0", shape=(2, 104), dtype=float32)

    with tf.Session() as sess:
        # 初始化全局变量
        sess.run(tf.global_variables_initializer())

        # 执行特征层转换
        transformed_features = sess.run(deep_input)
        print(transformed_features)

