#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

# new version grpc will check ENV http_proxy and https_proxy. so unset them first
unset http_proxy
unset https_proxy

pid_file=$SELFTF_FILE_AGENT_PID
out_file=$SELFTF_FILE_AGENT_OUT
if [ -f $pid_file ]; then
    kill `cat $pid_file`
fi

DST_IP=`ifconfig ${SELFTF_MAIN_NIC} | awk '/inet /{print substr($2, 1)}'`

if [ "${#DST_IP}" -eq "0" ]
then
  HOSTNAME=`hostname`
else
  HOSTNAME=`host $DST_IP | awk '{print $NF}' | cut -d . -f 1`
fi

NUMBER_OF_THREADS=$(grep -c ^processor /proc/cpuinfo)

$SCRIPT_PYTHON_EXECUTABLE selftf/agent.py \
--agent_workspace_path $SELFTF_HDFS_HOME \
--amqp_host ${AMPQ_MASTER_NODE} \
--hostname $HOSTNAME \
--agent_num_thread $NUMBER_OF_THREADS \
--agent_workspace_path "$SELFTF_HDFS_HOME" \
&> ${out_file} &

echo $! > $pid_file

echo `hostname` agent is started with pid $!
