# -*-coding:UTF-8-*-
from __future__ import print_function

import logging
import os

import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework
from selftf.tf_job.alexnet_imagenet.alexnet_model import classifier
import selftf.tf_job.alexnet_imagenet.alexnet_train_util as tu


class AlexNet_imagenet(MLJobFramework):
    def __init__(self):
        MLJobFramework.__init__(self)
        self.train_img_path = None
        self.num_features = 0

        self.wnid_labels = None
        self.observe_loss = None

        self.x = None
        self.y = None
        self.enqueue_op = None

        tf.app.flags.DEFINE_string("data_dir", "", "")
        tf.app.flags.DEFINE_integer("num_class", 1000, "")

    def model_definition(self, context):
        """
        :param selftf.lib.tf_program_util.TFProgramUtil context:
        :return:
        """
        num_class = context.FLAGS.num_class
        tu._class = num_class
        num_partition = context.get_n_partition()

        lmbda = 5e-04

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        batch_size = context.get_batch_size()
        context.set_global_step(global_step)
        self.wnid_labels = tu.get_winds(num_class, os.path.join(context.FLAGS.data_dir, "train"))

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, [224, 224, 3] -> image
            self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3],
                                    name="x-input")
            # target 1000 output classes
            self.y = tf.placeholder(tf.float32, shape=[None, num_class],
                                     name="y-input")
        # queue for image
        with tf.device(tf.no_op().device):
            q = tf.FIFOQueue(100, [tf.float32, tf.float32],
                             shapes=[[batch_size, 224, 224, 3], [batch_size, num_class]],
                             shared_name="imagenet_data_queue")
            self.enqueue_op = q.enqueue([self.x, self.y])
            x_b, y_b = q.dequeue()

        # creat an AlexNet
        pred, _ = classifier(x_b, 0.5, num_class, num_partition)

        # specify cost function
        # cross-entropy and weight decay
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_b,
                                                        name='cross-entropy'))

        with tf.name_scope('l2_loss'):
            l2_loss = tf.reduce_sum(lmbda * tf.stack(
                [tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
            tf.summary.scalar('l2_loss', l2_loss)

        with tf.name_scope('loss'):
            loss = cross_entropy + l2_loss
            tf.summary.scalar('loss', loss)

        context.set_train_op(loss=loss, sync=True)

    def has_enqueue_func(self):
        return True

    def get_enqueue_func(self, context, sess):
        """
        :param tf.Session sess:
        :param selftf.lib.tf_program_util.TFProgramUtil context:
        :return:
        """
        batch_size = context.get_batch_size()
        wind_labels = self.wnid_labels
        enqueue_op = self.enqueue_op
        x = self.x
        y = self.y

        def enqueue_func():
            x_b, y_b = tu.read_batch(batch_size, context.FLAGS.data_dir + "/train/",
                                     wind_labels)
            sess.run(enqueue_op, feed_dict={
                x: x_b,
                y: y_b
            }, options=tf.RunOptions(timeout_in_ms=60000))

        return enqueue_func


model = AlexNet_imagenet()
model.run()











































