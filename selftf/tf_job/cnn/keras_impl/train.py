import logging
import time
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple

from absl import app #, flags

from selftf.lib.mltuner.mltuner_util import MLTunerUtil
from selftf.lib.mltuner.mltuner_util import convert_model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

MODELS = {
    "vgg16": (
    tf.keras.applications.vgg16.preprocess_input, tf.keras.applications.VGG16),
    "vgg19": (
    tf.keras.applications.vgg16.preprocess_input, tf.keras.applications.VGG19),
    "inceptionv3": (tf.keras.applications.inception_v3.preprocess_input,
                    tf.keras.applications.InceptionV3),
    "xception": (tf.keras.applications.xception.preprocess_input,
                 tf.keras.applications.Xception),
    "resnet50": (tf.keras.applications.resnet.preprocess_input,
                 tf.keras.applications.ResNet50),
    "inceptionresnetv2": (
    tf.keras.applications.inception_resnet_v2.preprocess_input,
    tf.keras.applications.InceptionResNetV2),
    "mobilenet": (tf.keras.applications.mobilenet.preprocess_input,
                  tf.keras.applications.MobileNet),
    "mobilenetv2": (tf.keras.applications.mobilenet_v2.preprocess_input,
                  tf.keras.applications.MobileNetV2),
    "densenet121": (tf.keras.applications.densenet.preprocess_input,
                    tf.keras.applications.DenseNet121),
    "densenet169": (tf.keras.applications.densenet.preprocess_input,
                    tf.keras.applications.DenseNet169),
    "densenet201": (tf.keras.applications.densenet.preprocess_input,
                    tf.keras.applications.DenseNet201),
    "nasnetlarge": (tf.keras.applications.nasnet.preprocess_input,
                    tf.keras.applications.NASNetLarge),
    "nasnetmobile": (tf.keras.applications.nasnet.preprocess_input,
                     tf.keras.applications.NASNetMobile)
}

DATASET = ['cifar10']
# Begin defination of input argument
flags = tf.compat.v1.app.flags
flags.DEFINE_enum(
    "model",
    None,
    list(MODELS.keys()),
    "Name of the model to be run",
    case_sensitive=False)
flags.DEFINE_enum("dataset", 'cifar10', DATASET, "Name of the dataset",
                  case_sensitive=False)
# end defination of input argument

FLAGS = flags.FLAGS

def main(argv):
    prprocess_fn = MODELS[FLAGS.model][0]
    model_cls = MODELS[FLAGS.model][1]
    dataset_name = FLAGS.dataset

    split = tfds.Split.TRAIN
    dataset_info: Tuple[tf.data.Dataset, tfds.core.DatasetInfo] = tfds.load(
        dataset_name, split=split, as_supervised=True,
        with_info=True)
    info = dataset_info[1]
    batch_size = mltunerUtil.get_batch_size()
    learning_rate = mltunerUtil.get_learning_rate()
    num_class = info.features["label"].num_classes
    image_shape = info.features["image"].shape


    # Begin input
    def preprocess(image: tf.Tensor, label: tf.Tensor):
        image = tf.cast(image, tf.float32)
        image = prprocess_fn(image)

        return image, label


    def input_fn():
        dataset = tfds.load(dataset_name, split=split, as_supervised=True) \
            .shard(mltunerUtil.get_num_worker(),
                   mltunerUtil.get_worker_index())
        dataset = dataset.map(preprocess).cache().repeat().shuffle(batch_size * 20)
        dataset = dataset.batch(batch_size).prefetch(20)

        return dataset

    # End input

    # Begin model part
    tf.compat.v1.get_variable_scope().set_partitioner(
        tf.compat.v1.fixed_size_partitioner(mltunerUtil.get_num_ps()))

    model = model_cls(weights=None,
                      classes=num_class,
                      input_shape=image_shape)

    opt = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=mltunerUtil.get_learning_rate())
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=opt)
    # model.summary()

    # Whether get the model for classification
    if FLAGS.get_model:
        logging.info("start transfer model.")
        convert_model(model,FLAGS.script_path)
        return
    
    # model_dir = "{}/{}".format("/opt/tftuner", mltunerUtil.get_job_id())
    model_dir = "./model_dir"
    #model_dir = "{}/{}".format("hdfs://ssd2.dbg.private:8020/user/root/zm_hdfs", mltunerUtil.get_job_id())

    strategy = tf.distribute.experimental.ParameterServerStrategy()  # Can be config by env
    session_config = mltunerUtil.get_tf_session_config()
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=session_config,
                                    save_summary_steps=None,
                                    save_checkpoints_secs=None,
                                    model_dir=model_dir)

    # Begin convert keras to estimator
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, config=config)


    # End convert keras to estimator

    class LoggerHook(tf.estimator.SessionRunHook):
        """Logs loss and runtime."""

        def __init__(self):
            self.last_run_timestamp = time.time()

        def after_run(self, run_context, run_values):
            session: tf.Session = run_context.session
            loss, step = session.run([tf.compat.v1.get_collection("losses")[0],
                                      tf.compat.v1.get_collection("global_step_read_op_cache")[0]])
            logging.debug("step:{} loss:{}".format(step, loss))
            mltunerUtil.report_iter_loss(step, loss,
                                         time.time() - self.last_run_timestamp)
            self.last_run_timestamp = time.time()

    max_steps = mltunerUtil.get_trial_budget()
    if max_steps is None:
        max_steps = 2000 # default values
    # Begin training
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=max_steps,
                                        hooks=[LoggerHook()])
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

    # wait for chief ready?
    if not (mltunerUtil.is_chief() or mltunerUtil.is_ps()):
        time.sleep(1)
        if not tf.io.gfile.exists(model_dir):
            logging.debug("wait for chief init")
            time.sleep(1)

    tf.estimator.train_and_evaluate(keras_estimator,
                                    train_spec,
                                    eval_spec)
# End training

# print('Eval result: {}'.format(eval_result))

if __name__ == '__main__':
    mltunerUtil = MLTunerUtil()
    app.run(main)
