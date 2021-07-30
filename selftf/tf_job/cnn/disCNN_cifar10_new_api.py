import logging

import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework

from selftf.lib import tf_program_util
from selftf.tf_job.cnn import cifar10


class CNN(MLJobFramework):

    def __init__(self):
        # for dataset
        tf.app.flags.DEFINE_string("data_dir", "", "")

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        batch_size = context.get_batch_size()
        n_partition = context.get_n_partition()

        # count the number of global steps
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        context.set_global_step(global_step)

        # input images
        x, y_ = cifar10.distorted_inputs(batch_size, context.get_tf_flag().data_dir)

        # creat an CNN for cifar10
        y_conv = cifar10.inference(x, batch_size, n_partition)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # specify optimizer
        context.set_train_op(loss=loss)

        # accuracy
        # with tf.name_scope('Accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == "__main__":
    model = CNN()
    model.run()
