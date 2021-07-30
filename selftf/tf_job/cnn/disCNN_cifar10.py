# -*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import os
import time
import logging

from selftf.lib import tf_program_util
from selftf.tf_job.cnn import cifar10

# get the optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY_LEVEL'] = 'DEBUG'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                      datefmt='%m-%d %H:%M:%S',
                    )

# input flags
tf.app.flags.DEFINE_string("imagenet_path", 10, "ImageNet data path")
tf.logging.set_verbosity(tf.logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tftuner = tf_program_util.TFProgramUtil()
FLAGS = tf.app.flags.FLAGS

# config
batch_size = tftuner.get_batch_size()
# batch_size = 100
target_loss = FLAGS.target_loss

# cluster specification
parameter_servers = FLAGS.ps_list.split(',')
logging.info("parameter server list: " + str(parameter_servers))

workers = FLAGS.worker_list.split(',')
logging.info("worker list: " + str(workers))

n_PS = len(parameter_servers)
n_Workers = len(workers)

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
server_config = tftuner.get_tf_config_proto()

if FLAGS.job_name == "ps":
    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index)
    server.join()
elif FLAGS.job_name == "worker":
    server = tf.train.Server(
        cluster,
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index,
        config=server_config)
    # Between-graph replicationee
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # More to come on is_chief...
        is_chief = tftuner.get_is_chief()
        # count the number of global steps
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)

        # input images
        x, y_ = cifar10.distorted_inputs(batch_size)

        # creat an CNN for cifar10
        logging.debug("Number of partiton: " + str(tftuner.get_n_partition()))
        y_conv = cifar10.inference(x, batch_size, tftuner.get_n_partition())

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # specify optimizer
        with tf.name_scope('train'):
            train_op = tftuner.set_train_op(loss=loss, global_step=global_step)
        # accuracy
        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tftuner.do_pre_build_graph(global_step)
        sess_config = tftuner.get_tf_config_proto()

    saver = tf.train.Saver(sharded=True, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) + [global_step])

    # logging.debug("Logging" all queues: %s" % str(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)))

    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False
    cost = 10000.000
    step = 1
    tftuner.pre_recovery()

    try:
        while not tftuner.flag_end_process:
            # if first time enter do recovery
            sv = tftuner.get_monitored_training_session(global_step, saver=saver)

            logging.info("Open new tensorflow session")
            with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
                try:
                    logging.debug("Session is ready")
                    tftuner.post_recovery(global_step=global_step, sess=sess)
                    begin_time = time.time()
                    batch_time = time.time()
                    tftuner.pre_do_all_iteration(sess)
                    while not tftuner.should_stop_iteration(step, cost) and not tftuner.is_reconfig():
                        _, cost, step = sess.run([tftuner.get_train_op(), loss, global_step])
                        tftuner.post_do_iteration(steps=step, loss=cost, timestamp=time.time(),
                                                  duration=time.time() - batch_time)
                        batch_time = time.time()
                    total_time = time.time() - begin_time

                except (KeyboardInterrupt, SystemExit):
                    pass
                except:
                    logging.exception("Something Wrong")
                finally:
                    tftuner.post_do_all_iteration(sess)
                    sv.stop()
    except:
        pass
    finally:
        sys.exit(0)




    # with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
    #     try:
    #
    #         tftuner.post_recovery(global_step=global_step, sess=sess)
    #         begin_time = time.time()
    #         batch_time = time.time()
    #         tftuner.pre_do_all_iteration(sess)
    #         while not tftuner.should_stop_iteration(step, cost):
    #             _, cost, step = sess.run([tftuner.get_train_op(), loss, global_step])
    #             tftuner.post_do_iteration(steps=step, loss=cost, timestamp=time.time(), duration=time.time() - batch_time)
    #             batch_time = time.time()
    #         total_time = time.time() - begin_time
    #     except (KeyboardInterrupt, SystemExit):
    #         pass
    #     except:
    #         logging.exception("Something Wrong")
    #     finally:
    #         tftuner.post_do_all_iteration(sess)
    #         sv.stop()
    #         sys.exit(0)
