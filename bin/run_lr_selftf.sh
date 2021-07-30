source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "LR" 10 108 10 108 0.000024 4560 "bo" 29 4 "RMSProp" "selftf"