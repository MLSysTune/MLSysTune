source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "SVM" 10 108 10 108 0.00008 2590 "bo" 29 4 "SGD" "selftf"