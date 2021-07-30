import os

import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework

from selftf.lib import tf_program_util
from selftf.tf_job.cnn import cifar10
import alexnet_model as alexnet
import alexnet_train_util as tu

class AlexnetImagenet(MLJobFramework):

    def __init__(self):
        # for dataset
        tf.app.flags.DEFINE_string("data_dir", "", "")

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        lmbda = 0.00005
        imagenet_path = context.FLAGS.data_dir
        batch_size = context.get_batch_size()

        # count the number of global steps
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        context.set_global_step(global_step)

         # input images
        train_img_path = os.path.join(imagenet_path, 'ILSVRC2012_img_train')
        ts_size = tu.imagenet_size(train_img_path)
        num_batches = int(float(ts_size) / batch_size)

        wnid_labels, _ = tu.load_imagenet_meta(
            os.path.join(imagenet_path, 'data/meta.mat'))

        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, 1000])

        lr = tf.placeholder(tf.float32)
        keep_prob = tf.placeholder(tf.float32)

        # queue of examples being filled on the cpu
        with tf.device('/cpu:0'):
            q = tf.FIFOQueue(batch_size * 3, [tf.float32, tf.float32],
                             shapes=[[224, 224, 3], [1000]])
            enqueue_op = q.enqueue_many([x, y])

            x_b, y_b = q.dequeue_many(batch_size)

        pred, _ = alexnet.classifier(x_b, keep_prob)

        # cross-entropy and weight decay
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_b,
                                                        name='cross-entropy'))

        with tf.name_scope('l2_loss'):
            l2_loss = tf.reduce_sum(lmbda * tf.stack(
                [tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))

        with tf.name_scope('loss'):
            loss = cross_entropy + l2_loss

        context.set_train_op(loss=loss)

        # accuracy
        # with tf.name_scope('Accuracy'):
        #     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == "__main__":
    model = AlexnetImagenet()
    model.run()
