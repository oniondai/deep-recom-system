# coding: utf-8

import tensorflow as tf
from tensorflow import feature_column as fc


def wide_and_deep_model_fn(features, labels, mode, params):
    # wide部分
    with tf.variable_scope("wide_part", reuse=tf.AUTO_REUSE):
        wide_input = fc.input_layer(features, params["wide_part_feature_columns"])
        wide_logit = tf.layers.dense(wide_input, 1, name="wide_part_variables")

    # deep部分
    with tf.variable_scope("deep_part"):
        deep_input = fc.input_layer(features, params["deep_part_feature_columns"])
        net = deep_input
        for unit in params["hidden_units"]:
            net = tf.layers.dense(net, unit, activation=tf.nn.relu)
            if "dropout_rate" in params and 0.0 < params["dropout_rate"] < 1.0:
                net = tf.layers.dropout(net, params["dropout_rate"], training=(mode == tf.estimator.ModeKeys.TRAIN))
            if params["batch_norm"]:
                net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
        deep_logit = tf.layers.dense(net, 1)

    # 整体logit
    total_logit = tf.add(wide_logit, deep_logit, name="total_logit")

    # 1.定义PREDICT阶段行为
    prediction = tf.sigmoid(total_logit, name="prediction")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prediction
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    # -----定义完毕-----

    y = labels["Label"]
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=total_logit), name="loss")

    accuracy = tf.metrics.accuracy(labels=y, predictions=tf.to_float(tf.greater_equal(prediction, 0.5)))
    auc = tf.metrics.auc(labels=y, predictions=prediction)

    # 2.定义EVAL阶段行为
    metrics = {"eval_accuracy": accuracy, "eval_auc": auc}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # -----定义完毕-----

    wide_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wide_part')
    deep_part_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deep_part')

    # wide部分优化器op
    wide_part_optimizer = tf.train.FtrlOptimizer(learning_rate=params["wide_part_learning_rate"])
    wide_part_op = wide_part_optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(),
                                                var_list=wide_part_vars)

    # deep部分优化器op
    if params["deep_part_optimizer"] == 'Adam':
        deep_part_optimizer = tf.train.AdamOptimizer(learning_rate=params["deep_part_learning_rate"], beta1=0.9,
                                                     beta2=0.999, epsilon=1e-8)
    elif params["deep_part_optimizer"] == 'Adagrad':
        deep_part_optimizer = tf.train.AdagradOptimizer(learning_rate=params["deep_part_learning_rate"],
                                                        initial_accumulator_value=1e-8)
    elif params["deep_part_optimizer"] == 'RMSProp':
        deep_part_optimizer = tf.train.RMSPropOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'ftrl':
        deep_part_optimizer = tf.train.FtrlOptimizer(learning_rate=params["deep_part_learning_rate"])
    elif params["deep_part_optimizer"] == 'SGD':
        deep_part_optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["deep_part_learning_rate"])
    deep_part_op = deep_part_optimizer.minimize(loss=loss, global_step=None, var_list=deep_part_vars)

    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.group(wide_part_op, deep_part_op)

    # 3.定义TRAIN阶段行为
    assert mode == tf.estimator.ModeKeys.TRAIN
    # 待观测的变量
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if var.name == "wide_part/wide_part_variables/kernel:0":
            wide_part_dense_kernel = var
        if var.name == "wide_part/wide_part_variables/bias:0":
            wide_part_dense_bias = var

    # tensorboard收集
    tf.summary.scalar("train_accuracy", accuracy[1])
    tf.summary.scalar("train_auc", auc[1])
    tf.summary.histogram("wide_part_dense_kernel", wide_part_dense_kernel)
    tf.summary.scalar("wide_part_dense_kernel_l2_norm", tf.norm(wide_part_dense_kernel))

    # 训练log打印
    log_hook = tf.train.LoggingTensorHook(
        {
            "train_loss": loss,
            "train_auc_0": auc[0],
            "train_auc_1": auc[1],
            # "wide_logit": wide_logit,
            # "deep_logit": deep_logit,
            # "wide_part_dense_kernel": wide_part_dense_kernel,
            # "wide_part_dense_bias": wide_part_dense_bias,
            # "wide_part_dense_kernel_l2_norm": tf.norm(wide_part_dense_kernel)
        },
        every_n_iter=100
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
