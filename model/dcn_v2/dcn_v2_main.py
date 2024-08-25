# coding: utf-8

import sys
sys.path.append('../../tools')
import tensorflow as tf
import pandas as pd

from tfrecord_input import input_fn_tfr
from build_feature_columns import build_feature_columns
from dcn_v2_model import deep_cross_network_model_fn

# 定义输入参数
flags = tf.app.flags

# 训练参数
flags.DEFINE_string("model_dir", "./model_dir", "Directory where model parameters, graph, etc are saved")
flags.DEFINE_string("output_dir", "./output_dir", "Directory where pb file are saved")

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
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the deep part")
flags.DEFINE_boolean("batch_norm", True, "Perform batch normalization (True or False)")
flags.DEFINE_float("dropout_rate", 0, "Dropout rate")
flags.DEFINE_integer("num_cross_layer", 1, "Numbers of cross layer")
flags.DEFINE_string("exporter", "final", "Model exporter type")

FLAGS = flags.FLAGS

def main(unused_argv):
    """训练入口"""

    # 1.构建特征处理列.
    sparse_feature_columns, dense_feature_columns  = build_feature_columns()
    global total_feature_columns
    total_feature_columns = sparse_feature_columns + dense_feature_columns

    params = {
        "dense_feature_columns": dense_feature_columns,
        "category_feature_columns": sparse_feature_columns,
        'hidden_units': FLAGS.hidden_units.split(','),
        "num_cross_layer": FLAGS.num_cross_layer,
        "learning_rate": FLAGS.learning_rate,
    }

    # 2.创建estimator
    estimator = tf.estimator.Estimator(
        model_fn=deep_cross_network_model_fn,
        params=params,
        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
				      save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )

    # 3.创建train_spec
    # num_epochs=FLAGS.num_epochs,
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.train_data,
				      num_epochs=None,
				      shuffle=True,
				      batch_size=FLAGS.batch_size),
        max_steps=FLAGS.train_steps
    )

    # 4.创建feature_spec
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

    # 5.创建 eval_spec
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn_tfr(tfr_file_path=FLAGS.eval_data, num_epochs=FLAGS.num_epochs, shuffle=False, batch_size=FLAGS.batch_size),
        steps=None,
        exporters=exporters
    )

    # 6.开始训练
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
    with open('evaluate_metrics.result', 'w') as fout:
        for key in sorted(metrics):
            print('%s: %s' % (key, metrics[key]))
            fout.write("%s: %s\n" % (key, metrics[key]))

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
