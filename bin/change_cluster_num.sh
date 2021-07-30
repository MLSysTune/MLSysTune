#!/bin/bash

source $(dirname "$0")/common.sh
cd $SELFTF_HOME

JOB_ID=$1
NUM_OF_NODE=$2

python selftf/client.py --action=change_config $JOB_ID $NUM_OF_NODE