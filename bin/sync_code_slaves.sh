#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

SLAVES_FILE=$SELFTF_HOME/slaves
NUM_SLAVES=$(wc -l $SLAVES_FILE | awk '{print $1}')

cd $SELFTF_HOME

cat $SLAVES_FILE | xargs -i -P $NUM_SLAVES rsync -avz --delete --exclude=*.pyc --exclude "__pycache__/" --exclude "python2.7/" bin miniconda3 selftf requirements.txt {}:$SELFTF_HOME
