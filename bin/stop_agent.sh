#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

pid_file=${SELFTF_FILE_AGENT_PID}

pid=$(cat $pid_file)

kill $pid
if [ `hostname` -ne $SELFTF_MASTER_NODE ]; then
    killall -9 $SCRIPT_PYTHON_EXECUTABLE
fi
rm $pid_file

echo `hostname` agent PID:$pid  is killed

