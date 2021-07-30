source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "CNN" 10 108 10 108 0.0001 100 "bo" 10 7 "Adam" "dry_run"