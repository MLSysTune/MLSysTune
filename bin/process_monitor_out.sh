source $(dirname "$0")/common.sh
cd ${SELFTF_HOME}

OUT=`ls | grep *.csv`

cat ${SELFTF_HOME}/monitor.out \
| grep "Collected iteration statistic" |sed -n 's/.*local_iteration_time:\(.*\), timestamp:\(.*\), loss:\(.*\)/\1,\2,\3/p' \
| sort -t',' -k2 > $OUT


mv $OUT ${SELFTF_HOME}/zl_debug/lr_data
