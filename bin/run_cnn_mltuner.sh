source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "CNN" 3 3 10 108 0.0001 100 "bo" 28 9 "Adam" "mltuner"