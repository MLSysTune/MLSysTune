from __future__ import print_function

import json

import tensorflow as tf
import os
import sys
import logging

import selftf.lib.common
from selftf.lib import tuner

logging.basicConfig(level=logging.DEBUG)

# get the optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY_LEVEL'] = 'DEBUG'


class TFProgramUtil:

    def __init__(self):
        # input flags
        tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
        tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
        tf.app.flags.DEFINE_float("targted_accuracy", 0.5, "targted accuracy of model")
        tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")

        # <--  Add-on by pstuner -->
        tf.app.flags.DEFINE_string("ps_list", "", "")
        tf.app.flags.DEFINE_string("worker_list", "", "")
        tf.app.flags.DEFINE_string("working_dir", "", "")
        tf.app.flags.DEFINE_bool("is_chief", False, "")
        tf.app.flags.DEFINE_integer("max_iteration", 20, "")
        tf.app.flags.DEFINE_string("conf_dict", "{}", "")
        tf.app.flags.DEFINE_float("learning_rate", 0.01, "")

        self.FLAGS = tf.app.flags.FLAGS
        self.conf_dict = json.loads(self.FLAGS.conf_dict)

    def get_worker_list(self):
        return FLAGS.worker_list.split(',')

    def get_ps_list(self):
        return FLAGS.ps_list.split(',')

    def get_working_dir(self):
        return FLAGS.working_dir

    def get_is_chief(self):
        return FLAGS.get_is_chief

    def get_max_iteration(self):
        max_iteration = FLAGS.max_iteration
        if max_iteration == -1:
            return sys.maxsize
        else:
            return max_iteration

    def get_job_name(self):
        return FLAGS.job_name

    def get_task_index(self):
        return FLAGS.task_index

    def get_targted_accuracy(self):
        return FLAGS.targted_accuracy

    def _get_optimizer(self, optimizer, learning_rate):
        if optimizer == "SGD":
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == "Adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate)
        elif optimizer == "Adagrad":
            return tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == "Ftrl":
            return tf.train.FtrlOptimizer(learning_rate)
        elif optimizer == "Adam":
            return tf.train.AdamOptimizer(learning_rate)
        elif optimizer == "Momentum":
            return tf.train.MomentumOptimizer(learning_rate)
        elif optimizer == "RMSProp":
            return tf.train.RMSPropOptimizer(learning_rate)

    def get_optimizer(self):
        optimizer_str = self.FLAGS.optimizer
        learning_rate = self.FLAGS.learning_rate
        return self._get_optimizer(optimizer_str, learning_rate)

    def get_intra_op_parallelism_threads(self):
        return self.conf_dict[selftf.lib.common._intra_op_parallelism_threads]

    def get_inter_op_parallelism_threads(self):
        return self.conf_dict[selftf.lib.common._inter_op_parallelism_threads]

    def get_tf_flag(self):
        return self.FLAGS

    def get_tf_config_proto(self):
        return tf.ConfigProto(
            inter_op_parallelism_threads = self.get_inter_op_parallelism_threads(),
            intra_op_parallelism_threads = self.get_intra_op_parallelism_threads(),
            device_filters=["/job:ps", "/job:worker/task:%d" % self.get_task_index()]
        )


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(Shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.constant(0.0, shape=Shape)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_optimizer(optimizer, learning_rate):
    if optimizer == "SGD":
        return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "Adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == "Adagrad":
        return tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == "Ftrl":
        return tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == "Adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "Momentum":
        return tf.train.MomentumOptimizer(learning_rate)
    elif optimizer == "RMSProp":
        return tf.train.RMSPropOptimizer(learning_rate)


pstuner_util = TFProgramUtil()

FLAGS = pstuner_util.get_tf_flag()

max_iteration = pstuner_util.get_max_iteration()

parameter_servers = FLAGS.ps_list.split(',')
logging.info("parameter server list: "+ str(parameter_servers))

workers = FLAGS.worker_list.split(',')
logging.info("worker list: "+ str(workers))

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
# <-- END Add-on by pstuner -->
# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.01
targted_accuracy = FLAGS.targted_accuracy
empoch = 1000
Optimizer = FLAGS.optimizer

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # Graph define and init
        # END Graph define and init
        v = tf.get_variable("a", [1], dtype=tf.int32, initializer=tf.zeros_initializer)
        const = tf.get_variable("b", [1], dtype=tf.int32, initializer=tf.ones_initializer)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    if FLAGS.task_index == 0:
        is_chief = True
    else:
        is_chief = False

    variables_check_op = tf.report_uninitialized_variables()

    a = tf.assign(v, tf.add(v, const))

    sv = tf.train.Supervisor(is_chief=is_chief,
                             global_step=global_step)
    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False

    with sv.prepare_or_wait_for_session(server_grpc_url) as sess:
        while (not state):
            uninitalized_variables = sess.run(variables_check_op)
            if (len(uninitalized_variables.shape) == 1):
                state = True
        # runtime begin
        sess.run(a)
        print(sess.run(v))
        # runtime end
    sv.stop
