import time

import os

import sys

from selftf.lib import tf_program_util

step = 0
final_accuracy = 0.0
batch_count = 100
cost = 0.0
elapsed_time = 0.0
frequency = 1


util = tf_program_util.TFProgramUtil()

FLAGS = util.get_tf_flag()

# cluster specification
max_iteration = FLAGS.max_iteration
if max_iteration == -1:
    max_iteration = 5000
#mk chkpt
chk_dir = sys.argv[1][14:] +"/chkpt"
if not os.path.exists(chk_dir):
    os.makedirs(chk_dir)

if FLAGS.job_name == "ps":
    time.sleep(8000)
else:
    for step in range(1,max_iteration+1):
        i = step
        util.print_iteration_statistic(i,0.0,0.23,time.time())
        time.sleep(0.01)



