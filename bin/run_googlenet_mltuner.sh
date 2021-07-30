source $(dirname "$0")/common.sh

export KEY_ESTIMATION_FUNC="bo"
export ONLINE_NUM_ITER_PER_OS=1
export ONLINE_OS_SIZE=1
export OS_SIZE=1
export NUM_ITER_PER_OS=1

cd $SELFTF_HOME
sh bin/stop_monitor.sh
sh bin/stop_all_agent.sh

sleep 2

bin/start_monitor.sh
sh bin/start_all_agent.sh

sleep 2

mode="mltuner"
if [ "$#" -ne 0 ]; then
    mode="dry_run"
fi

python selftf/client.py \
--action submit_job \
--batch_size 100 \
--learning_rate 0.0001 \
--target_loss 0.2 \
--optimizer Adam \
--num_worker 35 \
--num_intra_thread 15 \
--mode $mode \
--script $SCRIPT_PYTHON_EXECUTABLE $SELFTF_HOME/selftf/tf_job/cnn/googlenet/train.py \
--data_dir=$DATASET_BASE_PATH/cifar-10-tfrecord
