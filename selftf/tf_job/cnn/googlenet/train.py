import os
import pickle

import numpy
import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework

from selftf.lib import tf_program_util
from selftf.lib.tf_program_util import TFProgramUtil
from selftf.tf_job.cnn import cifar10
from selftf.tf_job.cnn.googlenet.source.examples import loader
from selftf.tf_job.cnn.googlenet.source.src.dataflow.base import DataFlow
from selftf.tf_job.cnn.googlenet.source.src.helper.trainer import Trainer
from selftf.tf_job.cnn.googlenet.source.src.nets.googlenet import \
    GoogLeNet_cifar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

class CIFARDataFlow(DataFlow):

    def __init__(self, data_dir, num_worker, task_index, batch_size,
                 context:TFProgramUtil, reader_thread=4):

        self.batch_size = batch_size
        self.context = context

        tf_record_description = {
            'label': tf.FixedLenFeature([], tf.int64),
            'raw_image': tf.FixedLenFeature([], tf.string),
        }

        self._augment_flow = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=20.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=0.0,
            channel_shift_range=0.0,
            fill_mode='nearest',
            cval=0.0,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)

        def tf_record_parser(example_proto):
            ret_dict = tf.parse_example(example_proto, tf_record_description)
            return ret_dict


        d = tf.data.Dataset.list_files("{}/*".format(data_dir))
        d = d.shard(num_worker, task_index)
        d = d.repeat()
        d = d.interleave(tf.data.TFRecordDataset,
                         cycle_length=reader_thread, block_length=batch_size)
        d = d.batch(batch_size=batch_size)
        d = d.map(tf_record_parser, num_parallel_calls=reader_thread)
        d = d.prefetch(1)

        self.iter = d.make_one_shot_iterator()
        self.tf_op_iter_get_next = self.iter.get_next()

        self.batch_count = 0

        NUM_CIFAR10_TRAINING_IMAGE = 50000
        self.BATCH_PER_WORKER_EPOCH = NUM_CIFAR10_TRAINING_IMAGE / batch_size / num_worker


    def next_batch_dict(self):
        sess = self.context.sess
        self.batch_count += 1

        batch = sess.run(self.tf_op_iter_get_next)
        images = []
        for raw_image in batch["raw_image"]:
            images.append(pickle.loads(raw_image))

        batch_image = numpy.transpose(numpy.array(images).reshape((len(batch['raw_image']), 3, 32, 32)), (0,2,3,1))
        batch_label = numpy.array(batch["label"])

        self._augment_flow.fit(batch_image)

        batch_image, batch_label = \
        self._augment_flow.flow(batch_image, batch_label,
                                batch_size=self.batch_size)[0]

        return (batch_image, batch_label)

    @property
    def epochs_completed(self):
        return self.batch_count / self.BATCH_PER_WORKER_EPOCH


class GogoleNet(MLJobFramework):

    def __init__(self):
        # for dataset
        tf.app.flags.DEFINE_string("data_dir", "", "") # $DATASET_BASE/cifar-10-tfrecord
        self.train_data:DataFlow = None
        self.train_model:GoogLeNet_cifar = None

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        self.train_data = CIFARDataFlow(
            context.get_tf_flag().data_dir,
            context.get_num_worker(),
            context.get_worker_index(),
            context.get_batch_size(),
            context
        )

        # Create a training model
        self.train_model = GoogLeNet_cifar(
            n_channel=3, n_class=10, pre_trained_path=None,
            bn=True, wd=0, sub_imagenet_mean=False,
            conv_trainable=True, fc_trainable=True)
        self.train_model.create_train_model()

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(1),
                                      trainable=False)

        context.set_global_step(global_step)
        context.set_train_op(loss=self.train_model.get_loss())

    def get_feed_dict(self, context):
        batch_image, batch_label = self.train_data.next_batch_dict()
        return {
            self.train_model.image:batch_image,
            self.train_model.label:batch_label,
            self.train_model.lr:context.get_learning_rate(),
            self.train_model.keep_prob:1.,
        }
        #
        # # input images
        # x, y_ = cifar10.distorted_inputs(batch_size, context.get_tf_flag().data_dir)

        # specify optimizer


        # accuracy
        # with tf.name_scope('Accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == "__main__":
    model = GogoleNet()
    model.run()