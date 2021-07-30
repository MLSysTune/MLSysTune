#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

if [ -z "$OS_SIZE" ]; then
   OS_SIZE=10
fi

if [ -z "$NUM_ITER_PER_OS" ]; then
   NUM_ITER_PER_OS=60
fi

if [ -z "$ONLINE_OS_SIZE" ]; then
   ONLINE_OS_SIZE=100
fi

if [ -z "$ONLINE_NUM_ITER_PER_OS" ]; then
   ONLINE_NUM_ITER_PER_OS=200
fi

if [ -z "$KEY_ESTIMATION_FUNC" ]; then
   KEY_ESTIMATION_FUNC="bo"
fi

if [ -z "$BATCH_SIZE_RANGE" ]; then
   BATCH_SIZE_RANGE="[100,5000]"
fi

if [ -z "$LEARNING_RATE_RANGE" ]; then
   LEARNING_RATE_RANGE="[0.00001,0.0001]"
fi


pid_file=monitor.pid
if [ -f $pid_file ]; then
    kill `cat $pid_file`
fi

# systemctl restart rabbitmq-server

sleep 2

${SCRIPT_PYTHON_EXECUTABLE} selftf/monitor.py \
--amqp_host $AMPQ_MASTER_NODE \
--hostname `hostname` \
--tftuner_home $SELFTF_HOME \
--os_size $OS_SIZE \
--num_iter_per_os $NUM_ITER_PER_OS \
--online_reconfig_size $ONLINE_OS_SIZE \
--online_num_iter_per_os $ONLINE_NUM_ITER_PER_OS \
--estimation_func $KEY_ESTIMATION_FUNC \
--batch_size_range $BATCH_SIZE_RANGE