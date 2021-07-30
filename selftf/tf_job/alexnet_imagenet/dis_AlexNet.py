# -*-coding:UTF-8-*-
from __future__ import print_function
import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework
from selftf.tf_job.alexnet_imagenet.AlexNet import AlexNet
import selftf.tf_job.alexnet_imagenet.train_util as tu

class AlexNet_imagenet(MLJobFramework):
    def __init__(self):
        MLJobFramework.__init__(self)
        self.train_img_path = None
        self.num_features = 0

        self.y_ = None
        self.x = None
        self.wnid_labels = None
        self.observe_loss = None

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        self.wnid_labels = tu.load_imagenet_meta('hdfs://ssd02:8020/user/root/train_data/imagenet/meta_data.txt')

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, [224, 224, 3] -> image
            self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="x-input")
            # target 1000 output classes
            self.y_ = tf.placeholder(tf.float32, shape=[None, 1000], name="y-input")

        # creat an AlexNet
        y_conv, _ = AlexNet(self.x, 0.5)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))

        with tf.name_scope('train'):
            context.set_train_op(loss=loss, global_step=global_step)

    def get_feed_dict(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        batch_size = context.get_batch_size()
        """
        val_x, val_y = tu.read_validation_batch_V2(batch_size,
                                                   '/root/data/ILSVRC/Data/CLS-LOC/val/',
                                                    '/root/code/disAlexNet/val_10.txt')
        """

        val_x, val_y = tu.read_batch(batch_size, "hdfs://ssd02:8020/user/root/train_data/imagenet/train/", self.wnid_labels)
        return {self.x: val_x,
                self.y_: val_y,
               }


model = AlexNet_imagenet()
model.run()











































