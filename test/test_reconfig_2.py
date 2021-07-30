# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on one parameter server (ps), while the ops
are executed on two worker nodes by default. The TF sessions also run on the
worker node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.


A Testing program to test reconfiguration scheme2
2 ps and 1 worker
- run a step
- in second step do reconfiguration (move variables form PS A to PS B)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.graph_editor as tfge
import selftf.lib.tf_program_util
import selftf.lib.device_setter

# os.environ["GRPC_TRACE"] = "all"
# os.environ["GRPC_VERBOSITY"] = "DEBUG"
from selftf.lib.tuner import PSTunerConfiguration

flags = tf.app.flags
flags.DEFINE_string("data_dir", "MINIST-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", 0,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_string("ps_hosts", "localhost:2222,localhost:2223",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", "worker", "job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28


def modify_graph_for_reconfig():
  # Start Reconfig
  # # Reconfig op
  def generate_variable_map():
    ret = {}
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in var_list:
      ret[v.op.name] = "/job:ps/task:0"
    return ret

  context = selftf.lib.tf_program_util.SelfTFOptimizerContext(
      variable_map=generate_variable_map(),
      worker_device="/job:worker/task:0",
      master_ps_device="/job:ps/task:0",
      graph=tf.get_default_graph())

  training_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

  # For Static optimizer variables
  context.reallocate_static_ops()
  # END For static optimizer variables

  # For Trainable Variable
  wrapper_replica_variables = context.create_moved_trainable_variable_replica_wrapper()
  # replace read_op
  context.update_input_of_gradient_op(wrapper_replica_variables)

  # END For Trainable Variable

  context.update_apply_ops_inputs_phase1(wrapper_replica_variables)

  # change apply_op to new device

  context.reallocate_apply_grad_op(training_variables)

  context.update_device_for_optimizer_variables_in_tv()

  opt_init_op = tf.variables_initializer(context.get_optimizer_variables())
  # END For Optimizer variables in Trainable Variable

  context.clear_all_ops_collocate()


  worker_cache_assign_ops = wrapper_replica_variables.get_assign_op_to_worker_cache()
  dst_ps_variable_assign_ops = wrapper_replica_variables.get_assign_op_to_dst_ps()

  # END Reconfig
  return worker_cache_assign_ops, dst_ps_variable_assign_ops, opt_init_op

def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  # Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
    "ps": ps_spec,
    "worker": worker_spec})

  server0 = tf.train.Server(
      cluster, job_name="ps", task_index=0)
  server1 = tf.train.Server(
      cluster, job_name="ps", task_index=1)
  server = tf.train.Server(
      cluster, job_name="worker", task_index=0)

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU

  with tf.device(
      selftf.lib.device_setter.replica_device_setter(
          worker_device="/job:worker/task:0",
          cluster=cluster,
          master_ps_device="/job:ps/task:0",
          list_tf_name=["hid_w", "hid_b", "sm_w", "sm_b"]
      )):
    global_step = tf.get_variable("global_step",
                                  initializer=tf.constant(0),
                                  trainable=False)

    # Variables of the hidden layer
    hid_w = tf.get_variable("hid_w",
                            initializer=tf.truncated_normal(
                                    [IMAGE_PIXELS * IMAGE_PIXELS,
                                     FLAGS.hidden_units],
                                    stddev=1.0 / IMAGE_PIXELS),
                            partitioner=tf.fixed_size_partitioner(2)
                            )
    hid_b = tf.get_variable("hid_b",
                            # shape=[FLAGS.hidden_units],
                            initializer=tf.zeros([FLAGS.hidden_units]),
                            partitioner=tf.fixed_size_partitioner(2)
                            )

    # Variables of the softmax layer
    sm_w = tf.get_variable("sm_w",
                           # shape=[FLAGS.hidden_units, 10],
                           initializer=tf.truncated_normal(
                                   [FLAGS.hidden_units, 10],
                                   stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
                           partitioner=tf.fixed_size_partitioner(2)
                           )
    sm_b = tf.get_variable("sm_b",
                           # shape=[10],
                           initializer=tf.zeros([10]),
                           partitioner=tf.fixed_size_partitioner(2)
                           )

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate, 0.9)

    train_step = opt.minimize(cross_entropy, global_step=global_step)

    init_op = tf.global_variables_initializer()


  sv = tf.train.Supervisor(
      is_chief=is_chief,
      logdir="logdir/",
      init_op=init_op,
      recovery_wait_secs=1,
      global_step=global_step,
  )

  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

  # The chief worker (task_index==0) session will prepare the session,
  # while the remaining workers will wait for the preparation to complete.
  if is_chief:
    print("Worker %d: Initializing session..." % FLAGS.task_index)
  else:
    print("Worker %d: Waiting for session to be initialized..." %
          FLAGS.task_index)

  with sv.prepare_or_wait_for_session(server.target, config=sess_config) as sess:

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    # Training feed
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    train_feed = {x: batch_xs, y_: batch_ys}

    _, step = sess.run([train_step, global_step], feed_dict=train_feed)
    local_step += 1

    now = time.time()
    print("%f: Worker %d: training step %d done (global step: %d)" %
          (now, FLAGS.task_index, local_step, step))

  print("Close session, perform reconfiguration")
  # Reconfig phase 1 Modify graph
  op1, op2, op3 = modify_graph_for_reconfig()

  tf.summary.FileWriter('logdir/', tf.get_default_graph())

  # Open a new session
  with tf.Session(target=server.target) as sess:
    # Run the pre_train op
    sess.run(op1)
    sess.run(op2)
    sess.run(op3)

    # Run the train op
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    train_feed = {x: batch_xs, y_: batch_ys}


    _, step = sess.run([train_step, global_step], feed_dict=train_feed)
    local_step += 1

  # Move the dest_ps to regular variable
  # Start training

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # Validation feed
    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))

if __name__ == "__main__":
  tf.app.run()




