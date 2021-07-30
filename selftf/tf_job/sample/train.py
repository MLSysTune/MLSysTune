import time
import tensorflow as tf
import tensorflow_datasets as tfds

from selftf.lib.mltuner.mltuner_util import MLTunerUtil
from selftf.lib.mltuner.mltuner_util import convert_model


mltunerUtil = MLTunerUtil()

split = tfds.Split.TRAIN
batch_size = 16
learning_rate = 0.001
dataset_info = tfds.load('cifar10', split=split, as_supervised=True, with_info=True)
info = dataset_info[1]
num_class = info.features["label"].num_classes
image_shape = info.features["image"].shape

prprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
model_cls = tf.keras.applications.MobileNetV2

def preprocess(image: tf.Tensor, label: tf.Tensor):
    image = tf.cast(image, tf.float32)
    image = prprocess_fn(image)
    return image, label

def input_fn():
    num_worker = mltunerUtil.get_num_worker()
    worker_index = mltunerUtil.get_worker_index()
    dataset = tfds.load('cifar10', split=split, as_supervised=True) \
        .shard(num_worker, worker_index)
    dataset = dataset.map(preprocess).cache().repeat().shuffle(batch_size * 20)
    dataset = dataset.batch(batch_size).prefetch(20)

    return dataset

model = model_cls(weights=None,
                    classes=num_class,
                    input_shape=image_shape)

opt = tf.compat.v1.train.GradientDescentOptimizer(
    learning_rate=learning_rate)
model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=opt)

FLAGS = tf.compat.v1.app.flags.FLAGS
if FLAGS.get_model:
    convert_model(model, FLAGS.script_path)
    exit(0)

strategy = tf.distribute.experimental.ParameterServerStrategy()  # Can be config by env
session_config = mltunerUtil.get_tf_session_config()
config = tf.estimator.RunConfig(train_distribute=strategy,
                                session_config=session_config,
                                save_summary_steps=None,
                                save_checkpoints_secs=None,
                                model_dir="./model_dir")

# Convert keras to estimator
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, config=config)

class LoggerHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self):
        self.last_run_timestamp = time.time()

    def after_run(self, run_context, run_values):
        session: tf.Session = run_context.session
        loss, step = session.run([tf.compat.v1.get_collection("losses")[0],
                                    tf.compat.v1.get_collection("global_step_read_op_cache")[0]])
        mltunerUtil.report_iter_loss(step, loss,
                                        time.time() - self.last_run_timestamp)
        self.last_run_timestamp = time.time()


train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=20,
                                    hooks=[LoggerHook()])
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)
tf.estimator.train_and_evaluate(keras_estimator,
                                train_spec,
                                eval_spec)