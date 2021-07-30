#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

SLAVES_FILE=$SELFTF_HOME/slaves
NUM_SLAVES=$(wc -l $SLAVES_FILE | awk '{print $1}')

cat $SLAVES_FILE | xargs -i -P $NUM_SLAVES ssh {} sh $SELFTF_HOME/bin/stop_agent.sh
