# -*-coding:UTF-8-*-
from __future__ import print_function

import time
import os
import logging

import sys

from selftf.lib import tf_program_util
from selftf.tf_job.classification.ml_model import *
from selftf.tf_job.classification.read_libsvm_data import *

# log config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY_LEVEL'] = 'DEBUG'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

# input flags
tftuner = tf_program_util.TFProgramUtil()
FLAGS = tftuner.get_tf_flag()

# config

num_features = FLAGS.num_Features
batch_size = FLAGS.Batch_size
learning_rate = FLAGS.Learning_rate
targeted_loss = FLAGS.targeted_loss
Optimizer = FLAGS.optimizer
Epoch = FLAGS.Epoch

n_intra_threads = tftuner.get_intra_op_parallelism_threads()

# cluster specification
parameter_servers = FLAGS.ps_list.split(',')
logging.info("parameter server list: "+ str(parameter_servers))

workers = FLAGS.worker_list.split(',')
logging.info("worker list: "+ str(workers))

n_PS = len(parameter_servers)
n_Workers = len(workers)

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})
server_config = tftuner.get_tf_config_proto()

# start a server for a specific task
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

        # inout data
        # if FLAGS.ML_model == "SVM":
        trainset_files = [("/root/data/url_svmlight/Day%d" % i) + ".svm" for i in range(121)]
        # else:
        #     trainset_files=["/root/data/kdd12.tr"]
        logging.debug("Process dataset from: %s" % trainset_files)
        train_filename_queue = tf.train.string_input_producer(trainset_files)
        train_reader = tf.TextLineReader()
        train_data_line = train_reader.read(train_filename_queue)
        with tf.variable_scope('placeholder'):
            y_shape = 2
            if FLAGS.ML_model == "SVM":
                y_shape = 1
            y = tf.placeholder(tf.float32, [None, y_shape])
            sp_indices = tf.placeholder(tf.int64, name="sp_indices")
            shape = tf.placeholder(tf.int64, name="shape")
            ids_val = tf.placeholder(tf.int64, name="ids_val")
            weights_val = tf.placeholder(tf.float32, name="weights_val")

        with tf.variable_scope('parameter'):
            x_data = tf.SparseTensor(sp_indices, weights_val, shape)
            # x_data_SVM = tf.sparse_to_den        se(sp_indices, shape, weights_val)

        with tf.variable_scope('loss'):
            SVM_loss = SVMModel_with_linear(x_data, y, num_features)
            LR_loss, LR_loss_l2 = LogisticRegressionModel(x_data, y, num_features)

        # specify optimizer
        with tf.variable_scope('train'):
            grad_op = tftuner.get_optimizer()
            if FLAGS.ML_model == "SVM":
                LR_train_op = grad_op.minimize(LR_loss_l2, global_step=global_step)
                SVM_train_op = grad_op.minimize(SVM_loss, global_step=global_step)

        init_op = tf.global_variables_initializer()
        tftuner.do_pre_build_graph(global_step)
        sess_config = tftuner.get_tf_config_proto()

    saver = tf.train.Saver(sharded=True)
    sv = tftuner.get_monitored_training_session(saver=saver)

    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False

    tftuner.pre_recovery()

    with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config, start_standard_services=True) as sess:
        try:
            tftuner.post_recovery(sess=sess)

            begin_time = time.time()
            batch_time = time.time()
            cost = 1000000.0
            step = 0
            tftuner.pre_do_all_iteration(sess)

            def stop_criteria(step, cost):
                if tftuner.get_max_iteration() != sys.maxsize:
                    if step >= tftuner.init_global_step + tftuner.get_max_iteration():
                        return True
                    else:
                        return False
                if FLAGS.ML_model == "SVM" and step < 5000 and tftuner.get_max_iteration()==sys.maxsize:
                    return False
                return tftuner.should_stop_iteration(step, cost)

            while (not sv.should_stop()) and (not stop_criteria(step, cost)):#n_batches_per_epoch * Epoch
                label_one_hot, label, indices, sparse_indices, weight_list, read_count = read_batch(sess, train_data_line,
                                                                                                    batch_size)
                if FLAGS.ML_model == "LR":
                    _, cost, step = sess.run([LR_train_op, LR_loss, global_step], feed_dict={y: label_one_hot,
                                                                                             sp_indices: sparse_indices,
                                                                                             shape: [read_count,
                                                                                                     num_features],
                                                                                             ids_val: indices,
                                                                                             weights_val: weight_list})
                else:
                    _, cost, step = sess.run([SVM_train_op, SVM_loss, global_step], feed_dict={y: label,
                                                                                               sp_indices: sparse_indices,
                                                                                               shape: [read_count,
                                                                                                       num_features],
                                                                                               ids_val: indices,
                                                                                               weights_val: weight_list})

                duration = time.time() - batch_time

                # re = str(step + 1) + "," + str(n_Workers) + "," + str(n_intra_threads) + "," + str(cost) + "," + str(
                #     duration)
                # process = open("/root/ex_result/baseline/" + FLAGS.ML_model + "_process.csv", "a+")
                # process.write(re + "\r\n")
                # process.close()

                # print("Step: %d," % (step + 1),
                #       " Loss: %f" % cost,
                #       " Bctch_Time: %fs" % float(duration))
                tftuner.post_do_iteration(steps=step, loss=cost, timestamp=time.time(), duration=duration)

                batch_time = time.time()
            # final_re = str(step + 1) + "," + str(n_Workers) + "," + str(n_intra_threads) + "," + str(cost) + "," + str(
            #     float(time.time() - begin_time))
            # result = open("/root/ex_result/baseline/" + FLAGS.ML_model + "_result.csv", "a+")
            # result.write(final_re + "\r\n")
            # result.close()
        except (KeyboardInterrupt, SystemExit):
            pass
        except :
            logging.exception("Something Wrong")
        finally:
            tftuner.post_do_all_iteration(sess)
            sv.stop()
            sys.exit(0)
