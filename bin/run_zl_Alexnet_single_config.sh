source $(dirname "$0")/common.sh

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
--batch_size 256 \
--learning_rate 0.01 \
--target_loss 0.5 \
--optimizer Adam \
--num_worker 24 \
--num_intra_thread 13 \
--n_partition 1 \
--mode dry_run \
--script /root/anaconda2/bin/python $SELFTF_HOME/selftf/tf_job/Alexnet_zl/disAlexnet.py