source $(dirname "$0")/common.sh

export KEY_ESTIMATION_FUNC="bo"
export ONLINE_NUM_ITER_PER_OS=500
export ONLINE_OS_SIZE=100
export OS_SIZE=10
export NUM_ITER_PER_OS=60

cd $SELFTF_HOME
sh bin/stop_monitor.sh
sh bin/stop_all_agent.sh

sleep 2

bin/start_monitor.sh
sh bin/start_all_agent.sh

sleep 2

python selftf/client.py \
--action submit_job \
--ml_model CNN \
--target_loss 0.5 \
--mode test_reconfig_2 \
--script $SCRIPT_PYTHON_EXECUTABLE $SELFTF_HOME/selftf/tf_job/cnn/disCNN_cifar10_new_api.py \
--data_dir=$DATASET_BASE_PATH/cifar-10-batches-bin/