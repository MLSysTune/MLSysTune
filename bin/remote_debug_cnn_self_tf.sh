# Assume monitor.py is started in debug

source $(dirname "$0")/common.sh

_CNN_batch_range = "(100,1000)"
_CNN_learning_rate_range = "(0.00001,0.0001)"

export KEY_ESTIMATION_FUNC="bo"
export ONLINE_NUM_ITER_PER_OS=500
export ONLINE_OS_SIZE=100
export OS_SIZE=20
export NUM_ITER_PER_OS=100
export LEARNING_RATE_RANGE=$_CNN_learning_rate_range
export BATCH_SIZE_RANGE=$_CNN_batch_range


cd $SELFTF_HOME
sh bin/stop_monitor.sh
sh bin/stop_all_agent.sh

sleep 2

# bin/start_monitor.sh
sh bin/start_all_agent.sh

sleep 2

python selftf/client.py \
--action submit_job \
--ml_model CNN \
--batch_size 100 \
--learning_rate 0.0001 \
--target_loss 0.5 \
--script $SCRIPT_PYTHON_EXECUTABLE $SELFTF_HOME/selftf/tf_job/cnn/disCNN_cifar10_new_api.py \
--data_dir=$DATASET_BASE_PATH/cifar-10-batches-bin/