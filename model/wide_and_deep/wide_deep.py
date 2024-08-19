# coding: utf-8
import tensorflow as tf
from tensorflow import feature_column as fc
import pandas as pd
# 定义输入参数
flags = tf.app.flags

# 训练参数
flags.DEFINE_string("model_dir", "./model_dir_wide_deep", "Directory where model parameters, graph, etc are saved")
flags.DEFINE_string("output_dir", "./output_dir", "Directory where pb file are saved")

# flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("train_data", "/data/deep-recom-system/data/criteo_x4/train.tr.tfrecords", "Path to the train data")
flags.DEFINE_string("eval_data", '/data/deep-recom-system/data/criteo_x4/valid.tr.tfrecords',
                    "Path to the evaluation data")
flags.DEFINE_string("test_data", '/data/deep-recom-system/data/criteo_x4/test.tr.tfrecords',
                    "Path to the evaluation data")
flags.DEFINE_integer("num_epochs", 2, "Epoch of training phase")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "Dataset shuffle buffer size")
flags.DEFINE_integer("num_parallel_readers", -1, "Number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 100, "Save checkpoints every this many steps")

# 模型参数
flags.DEFINE_integer("batch_size", 32, "Training batch size")
flags.DEFINE_float("wide_part_learning_rate", 0.005, "Wide part learning rate")
flags.DEFINE_float("deep_part_learning_rate", 0.001, "Deep part learning rate")
flags.DEFINE_string("deep_part_optimizer", "Adam",
                    "Wide part optimizer, supported strings are in {'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'}")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0, "Dropout rate")
flags.DEFINE_string("exporter", "final", "Model exporter type")

FLAGS = flags.FLAGS


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
        #linear_feature_columns.append(fc.categorical_column_with_identity(feat, 1000))

    # 对于dense特征，使用原值.
    for feat in dense_features:
        #dnn_feature_columns.append(fc.numeric_column(feat))
        linear_feature_columns.append(fc.numeric_column(feat))

    return dnn_feature_columns, linear_feature_columns


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

    # 总体logit
    total_logit = tf.add(wide_logit, deep_logit, name="total_logit")

    # -----定义PREDICT阶段行为-----
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

    # -----定义EVAL阶段行为-----
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

    # -----定义TRAIN阶段行为-----
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

def main(unused_argv):
    """训练入口"""

    deep_columns, wide_columns  = build_feature_columns()
    global total_feature_columns
    total_feature_columns = wide_columns + deep_columns

    params = {
        "wide_part_feature_columns": wide_columns,
        "deep_part_feature_columns": deep_columns,
        'hidden_units': FLAGS.hidden_units.split(','),
        "dropout_rate": FLAGS.dropout_rate,
        "batch_norm": FLAGS.batch_norm,
        "deep_part_optimizer": FLAGS.deep_part_optimizer,
        "wide_part_learning_rate": FLAGS.wide_part_learning_rate,
        "deep_part_learning_rate": FLAGS.deep_part_learning_rate,
    }

    estimator = tf.estimator.Estimator(
        model_fn=wide_and_deep_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
				      save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    # num_epochs=FLAGS.num_epochs,
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.train_data,
				      num_epochs=None,
				      shuffle=True,
				      batch_size=FLAGS.batch_size),
        max_steps=FLAGS.train_steps
    )

    feature_spec = tf.feature_column.make_parse_example_spec(total_feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    if FLAGS.exporter == "latest":
        exporter = tf.estimator.LatestExporter(
            name='latest_exporter',
            serving_input_receiver_fn=serving_input_receiver_fn)
    elif FLAGS.exporter == "best":
        exporter = tf.estimator.BestExporter(
            name='best_exporter',
            serving_input_receiver_fn=serving_input_receiver_fn,
	    exports_to_keep=5)
    else:
        exporter = tf.estimator.FinalExporter(
            name='final_exporter',
            serving_input_receiver_fn=serving_input_receiver_fn)
    exporters = [exporter]

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.eval_data, num_epochs=FLAGS.num_epochs, shuffle=False, batch_size=FLAGS.batch_size),
        steps=None,
        exporters=exporters
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # 显式的将模型保存在本地, export也可以将模型保存在本地.
    print("exporting model ...")
    estimator.export_savedmodel(FLAGS.output_dir, serving_input_receiver_fn)

    # 打印训练后的指标estimator.evaluate.
    print("Start Evaluate Metrics")
    metrics = estimator.evaluate(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.eval_data,
        			      num_epochs=1,
        			      shuffle=False,
        			      batch_size=32))
    for key in sorted(metrics):
        print('%s: %s' % (key, metrics[key]))

    # 打印预测结果 estimator.predict
    results = estimator.predict(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.eval_data,
        			      num_epochs=1,
        			      shuffle=False,
        			      batch_size=32))
    predicts_df = pd.DataFrame.from_dict(results)
    predicts_df['probabilities'] = predicts_df['probabilities'].apply(lambda x: x[0])
    test_df = pd.read_csv("/data/deep-recom-system/data/criteo_x4/valid.csv_demo")
    predicts_df['Label'] = test_df['Label']
    predicts_df.to_csv("predictions.csv")
    print("after evaluate")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
