import logging
import os
import random
import time

import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework

from selftf.lib import tf_program_util
from selftf.tf_job.classification.ml_model import SVMModel_with_linear, LogisticRegressionModel
from selftf.tf_job.classification.read_libsvm_data import read_batch

class SVM_LR(MLJobFramework):

    def __init__(self):
        MLJobFramework.__init__(self)
        self.tf_op_iter_get_next = None
        self.train_data_batch_tensor = None
        self.num_features = 0

        self.y = None
        self.sp_indices = None
        self.shape = None
        self.ids_val = None
        self.weights_val = None

        self.observe_loss = None

        self.enqueue_ops = None
        # for dataset
        tf.app.flags.DEFINE_string("data_dir", "", "")

    def data_input_definition(self, num_worker, worker_index, path, batch_size,
                              reader_thread=4):

        d = tf.data.Dataset.list_files("{}/*".format(path))
        d = d.shard(num_worker, worker_index)
        d = d.repeat()
        d = d.shuffle(batch_size*3)
        d = d.interleave(tf.data.TextLineDataset,
                         cycle_length=reader_thread,
                         block_length=batch_size)

        d = d.batch(batch_size=batch_size)
        d = d.prefetch(1)

        tf_iter = d.make_one_shot_iterator()
        self.tf_op_iter_get_next = tf_iter.get_next()

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        FLAGS = context.get_tf_flag()

        self.num_features = FLAGS.num_Features
        path = context.get_tf_flag().data_dir
        batch_size = context.get_batch_size()

        # count the number of global steps
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        context.set_global_step(global_step)
        batch_size = context.get_batch_size()

        # data loader
        self.data_input_definition(context.get_num_worker(),
                                   context.get_worker_index(),
                                   path,
                                   batch_size)

        with tf.variable_scope('placeholder'):
            if FLAGS.ML_model == "SVM":
                y_shape = 1
            else:
                y_shape = 2
            self.y = tf.placeholder(tf.float32, [batch_size, y_shape])
            self.sp_indices = tf.placeholder(tf.int64, name="sp_indices")
            self.shape = tf.placeholder(tf.int64, name="shape")
            self.ids_val = tf.placeholder(tf.int64, name="ids_val")
            self.weights_val = tf.placeholder(tf.float32, name="weights_val")

        x_data = tf.SparseTensor(self.ids_val, self.weights_val, dense_shape=[batch_size, self.num_features])
        num_qr_before_shuffle_batch = len(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
        batch_x_data, batch_label = tf.train.shuffle_batch(
            [x_data, self.y],
            batch_size=1,
            capacity=100,
            min_after_dequeue=0,
            shared_name="data_queue"
        )

        # Goal: get enqueue func
        assert len(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)) - num_qr_before_shuffle_batch == 1
        queue_runner = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[-1]  # type: tf.train.QueueRunner
        self.enqueue_ops = queue_runner.enqueue_ops
        # Goal : to remove the queue runner op added by train_data_batch_tensor()
        qr_list = tf.get_collection_ref(tf.GraphKeys.QUEUE_RUNNERS)
        qr_list.remove(qr_list[-1])

        # # Goal replace enqueues queue to a dummy queue to prevent the real queue being closed
        # with tf.device(tf.no_op().device):
        #     dummy_queue = tf.FIFOQueue(1, tf.int32, [1])
        # assert len(self.enqueue_ops) == 1
        # enqueue_op = self.enqueue_ops[0]
        # enqueue_op._update_input(0, dummy_queue.queue_ref)

        batch_x_data = tf.sparse_reshape(batch_x_data,
                                         [batch_size, self.num_features])

        with tf.variable_scope('loss'):
            if FLAGS.ML_model == "SVM":
                batch_label = tf.reshape(batch_label, [batch_size,1])
                SVM_loss = SVMModel_with_linear(batch_x_data, batch_label, self.num_features)
                self.observe_loss = SVM_loss
            else:
                batch_label = tf.reshape(batch_label, [batch_size,2])
                LR_loss, LR_loss_l2 = LogisticRegressionModel(batch_x_data, batch_label, self.num_features)
                self.observe_loss = LR_loss

        # specify optimizer
        if FLAGS.ML_model == "SVM":
            # LR_train_op = grad_op.minimize(LR_loss_l2, global_step=global_step)
            context.set_train_op(SVM_loss)
        else:
            # SVM_train_op = grad_op.minimize(SVM_loss, global_step=global_step)
            context.set_train_op(LR_loss_l2)

    def has_enqueue_func(self):
        return True

    def get_enqueue_func(self, context, sess):
        """
        :param tf.Session sess:
        :param selftf.lib.tf_program_util.TFProgramUtil context:
        :return:
        """
        enqueue_ops = self.enqueue_ops
        placeholder_ids = self.ids_val
        placeholder_weight = self.weights_val
        placeholder_label = self.y

        if context.FLAGS.ML_model == "SVM":
            is_one_hot = False
        elif context.FLAGS.ML_model == "LR":
            is_one_hot = True

        def enqueue_func():
            """
            :type tftuner: tf_program_util.TFProgramUtil
            """
            buffer = sess.run(self.tf_op_iter_get_next)
            label, indices, values = read_batch(buffer, is_one_hot)
            sess.run(enqueue_ops,
                     feed_dict={
                         placeholder_ids: indices,
                         placeholder_weight: values,
                         placeholder_label: label
                     },
                     options=tf.RunOptions(
                        timeout_in_ms=int(10*1000*2)
                     ))

        return enqueue_func

    def get_observe_loss_variable(self, context):
        return self.observe_loss


model = SVM_LR()
model.run()
