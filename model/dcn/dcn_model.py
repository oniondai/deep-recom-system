# coding: utf-8

import tensorflow as tf
from tensorflow import feature_column as fc

def cross_layer(x0, xl, index):
    """
        dcn模型的cross layer
    Args:
        x0 (tensor): cross layer最原始输入
        xl (tensor): cross layer上一层输出
        index (int): cross layer序号

    Returns:
        tensor, 维度与x0一致
    """

    dimension = int(x0.get_shape()[-1])
    # wl，bl为cross_layer参数
    wl_name = "wl_%s" % index
    bl_name = "bl_%s" % index
    wl = tf.get_variable(name=wl_name, shape=(dimension, 1), dtype=tf.float32)  # (d, 1)
    bl = tf.get_variable(name=bl_name, shape=(dimension, 1), dtype=tf.float32)  # (d, 1)

    xl_wl = tf.matmul(xl, wl)  # (batch, d) * (d, 1) = (batch, 1)
    x0_xl_wl = tf.multiply(x0, xl_wl)   # (batch, d) multiply (batch, 1) = (batch, d)
    output = tf.add(x0_xl_wl, tf.transpose(bl))
    output = tf.add(output, xl)

    return output

def deep_cross_network_model_fn(features, labels, mode, params):
    # 连续特征
    with tf.variable_scope("dense_input"):
        dense_input = fc.input_layer(features, params["dense_feature_columns"])

    # 类别特征
    with tf.variable_scope("category_input"):
        category_input = fc.input_layer(features, params["category_feature_columns"])

    concat_all = tf.concat([dense_input, category_input], axis=-1)

    with tf.variable_scope("cross_part"):
        cross_vec = concat_all
        for i in range(params["num_cross_layer"]):
            cross_vec = cross_layer(x0=concat_all, xl=cross_vec, index=i)

    with tf.variable_scope("dnn_part"):
        dnn_vec = concat_all
        for i, unit in enumerate(params["hidden_units"]):
            dnn_name = "dnn_dense_%s" % i
            dnn_vec = tf.layers.dense(dnn_vec, unit, activation=tf.nn.relu, name=dnn_name)

    with tf.variable_scope("output_part"):
        output = tf.concat([cross_vec, dnn_vec], axis=-1)
        logit = tf.layers.dense(output, 1, activation=None)

    # 1.定义PREDICT阶段行为
    prediction = tf.sigmoid(logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "logit": logit,
            'probabilities': prediction,
        }
        saved_model_output = {
            'probabilities': prediction,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(saved_model_output)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    y = labels["Label"]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit), name="loss")

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # 2.定义EVAL阶段行为
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"], beta1=0.9,
                                       beta2=0.999, epsilon=1e-8)
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    # 3.定义TRAIN阶段行为
    assert mode == tf.estimator.ModeKeys.TRAIN

    # tensorboard收集
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])

    # 训练log打印
    log_hook = tf.train.LoggingTensorHook(
        {
            "train_loss": loss,
            "train_auc_0": auc[0],
            "train_auc_1": auc[1],
            #"logit": logit,
        },
        every_n_iter=100
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
