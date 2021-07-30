source $(dirname "$0")/common.sh

cd $SELFTF_HOME

/usr/bin/python selftf/test_os_parameter.py "ALEXNET_IMAGENET" 10 10 10 108 0.02 32 "bo" 17 10 "RMSProp_Imagenet" "dry_run"