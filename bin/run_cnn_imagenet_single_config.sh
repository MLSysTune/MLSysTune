source $(dirname "$0")/common.sh

export KEY_ESTIMATION_FUNC="bo"
export ONLINE_NUM_ITER_PER_OS=500
export ONLINE_OS_SIZE=100
export OS_SIZE=20
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
--batch_size 128 \
--learning_rate 0.001 \
--target_loss 0.05 \
--optimizer Adam \
--num_worker 24 \
--num_intra_thread 13 \
--n_partition 1 \
--mode dry_run \
--script /root/anaconda2/bin/python $SELFTF_HOME/selftf/tf_job/cnn/AlexNet/dis_AlexNet.py
