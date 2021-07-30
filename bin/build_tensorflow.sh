source $(dirname "$0")/common.sh

cd ${SELFTF_HOME}
SOURCE_PATH=/root/tensorflow
SELFTF_PATH=$SELFTF_HOME
cd $SOURCE_PATH

rm -rf /tmp/tensorflow*.whl

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/

WHL=$(ls /tmp/tensorflow*.whl)

mv ${WHL} ${SELFTF_HOME}

cat $SELFTF_PATH/slaves | xargs -i -P40 ssh {} bash -c "pip install --upgrade `ls ${SELFTF_HOME}/tensorflow*.whl`"

rm ${SELFTF_HOME}/tensorflow*.whl
