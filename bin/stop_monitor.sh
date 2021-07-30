#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

pid_file=monitor.pid

pid=$(cat $pid_file)

kill -9 $pid
rm $pid_file

echo "monitor PID: $pid is killed"

