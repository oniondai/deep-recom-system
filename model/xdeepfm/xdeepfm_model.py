# coding: utf-8

import tensorflow as tf
from tensorflow import feature_column as fc

def cin_layer(x0, xk, hk_1, index):
    """
        xdeepfm模型的Compressed Interaction Network部分
    Args:
        x0 (tensor): 原始输入tensor, shape=(batch, m, D)
        xk (tensor): 上一CIN层输出tensor, shape=(batch, hk, D)
        hk_1 (int): 超参数卷积核的大小，也是本层输出维度, 也是feature map个数
        index (int): 序号

    Returns:
        tensor, 维度与输入tensor一致
    """

    D = int(x0.get_shape()[-1])
    m = int(x0.get_shape()[1])
    hk = int(xk.get_shape()[1])

    outer = tf.einsum('...ik,...jk -> ...kij', xk, x0)  # (batch, D, hk, m)
    outer = tf.reshape(outer, shape=(-1, D, hk*m))  # (batch, D, hk*m)

    # 卷积核, 相当于论文中的wk
    cin_layer_idx_filter =  "cin_layer_%s_filter" % index
    filters = tf.get_variable(name=cin_layer_idx_filter, shape=(1, hk*m, hk_1))
    # 一维卷积操作
    xk_1 = tf.nn.conv1d(outer, filters=filters, stride=1, padding="VALID")  # (batch, D, hk_1)
    xk_1 = tf.transpose(xk_1, perm=[0, 2, 1])   # (batch, hk_1, D)

    return xk_1

def xdeepfm_model_fn(features, labels, mode, params):
    # 连续特征
    with tf.variable_scope("dense_input"):
        dense_input = fc.input_layer(features, params["dense_feature_columns"])

    # 类别特征
    with tf.variable_scope("category_input"):
        category_input = fc.input_layer(features, params["category_feature_columns"])  # (batch, m*D)

    # 线性部分
    with tf.variable_scope("linear_part"):
        linear_vec = tf.concat([dense_input, category_input], axis=-1)
        linear_logit = tf.layers.dense(linear_vec, 1, activation=None, use_bias=True)  # (batch, 1)

    # CIN部分
    with tf.variable_scope("cin_part"):
        x0 = tf.reshape(category_input, shape=(-1, len(params["category_feature_columns"]), params["embedding_dim"])) # (batch, m, D), m指的是特征的个数，D指的是embed的维度，这里是(bs, 26, 4)
        xk = x0
        x_container = []    # [(batch, h1, D), (batch, h2, D), ...], h指的是卷积核的大小，由cin_layer_feature_maps设置
        # 卷积核参数h 越大，模型能够捕捉的高阶特征交互就越丰富，同时意味着更多的参数和更复杂的计算
        for i, features_map_num in enumerate(params["cin_layer_feature_maps"]):
            xk = cin_layer(x0, xk, features_map_num, i+1)
            x_container.append(xk) # 这里输出是(batch, h1, D)，卷积核h的取值经验是[64, 32, 16] 
        p_plus = [tf.reduce_sum(x, axis=-1) for x in x_container]   # [(batch, h1), (batch, h2), ...]
        p_plus = tf.concat(p_plus, axis=-1)  # (batch, Σhi)
        cin_logit = tf.layers.dense(p_plus, 1, activation=None, use_bias=False)  # (batch, 1)

    # dnn部分
    with tf.variable_scope("dnn_part"):
        dnn_vec = linear_vec
        for i, unit in enumerate(params["hidden_units"]):
            dense_idx = "dense_%s" % i
            dnn_vec = tf.layers.dense(dnn_vec, unit, activation=tf.nn.relu, name=dense_idx)
        dnn_logit = tf.layers.dense(dnn_vec, 1, activation=None, use_bias=False)  # (batch, 1)

    # 合并
    logit = tf.add_n([linear_logit, cin_logit, dnn_logit])
    #logit = linear_logit + cin_logit + dnn_logit

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
            #"linear_logit": linear_logit,
            #"cin_logit": cin_logit,
            #"dnn_logit": dnn_logit,
        },
        every_n_iter=100
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[log_hook])
