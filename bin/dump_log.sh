source $(dirname "$0")/common.sh

cd $SELFTF_HOME

python selftf/client.py \
--action dump_log $1