# coding: utf-8
import tensorflow as tf
from tensorflow import feature_column as fc

# 特征定义
def build_feature_columns():
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    sparse_feature_columns = []
    dense_feature_columns = []

    # 对于sparse类别特征，转成embed.
    for i, feat in enumerate(sparse_features):
        sparse_feature_columns.append(fc.embedding_column(
            fc.categorical_column_with_hash_bucket(feat, 1000), 4))
        #dense_feature_columns.append(fc.categorical_column_with_identity(feat, 1000))

    # 对于dense特征，使用原值.
    for feat in dense_features:
        #sparse_feature_columns.append(fc.numeric_column(feat))
        dense_feature_columns.append(fc.numeric_column(feat))

    return sparse_feature_columns, dense_feature_columns
