import tensorflow as tf
import pandas as pd
from tqdm import tqdm

def make_example(row, sparse_feature_name, dense_feature_name, label_name):
    features = {}
    # 类别sparse特征一般是字符串特征，使用tf.train.BytesList
    for category_feature in sparse_feature_name:
        category_value = row[category_feature] if not pd.isna(row[category_feature]) else ''
        features[category_feature] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[category_value.encode()])
         )

    # 数值dense特征一般是数值，使用tf.train.FloatList
    for dense_feature in dense_feature_name:
        features[dense_feature] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[row[dense_feature]])
        )

    # 标签特征一般是数值，使用tf.train.FloatList
    for label in label_name:
        features[label] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[row[label]])
        )

    return tf.train.Example(features=tf.train.Features(feature=features))

def write_tfrecord(tfr_file, df, sparse_feature_names, dense_feature_names, label_name):
    writer = tf.python_io.TFRecordWriter(tfr_file)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        ex = make_example(row, sparse_feature_names, dense_feature_names, label_name)
        writer.write(ex.SerializeToString())
    writer.close()

if __name__ == "__main__":
    data_path = './criteo_x4/'
    train_data = 'train.csv_demo'
    valid_data = 'valid.csv_demo'
    test_data = 'test.csv_demo'

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['Label']

    for data in [train_data, valid_data, test_data]:
        data_file = data_path + data
        data = pd.read_csv(data_file)
        # 对sparse缺失的补空字符串
        data[sparse_features] = data[sparse_features].fillna('', )
        # 对dense缺失的补0
        data[dense_features] = data[dense_features].fillna(0, )

        out_tfr = "%s.tr.tfrecords" % (data_file.split(".csv")[0])
        write_tfrecord(out_tfr, data, sparse_features, dense_features, target)
