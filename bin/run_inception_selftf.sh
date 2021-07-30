source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "INCEPTION" 10 10 10 108 0.00005 10 "bo" 29 4 "RMSProp" "selftf"