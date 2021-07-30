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
--ml_model SVM \
--batch_size 2700 \
--learning_rate 0.00007 \
--target_loss 0.1 \
--optimizer Momentum \
--num_worker 19 \
--num_intra_thread 5 \
--mode test_reconfig_2 \
--script $SCRIPT_PYTHON_EXECUTABLE $SELFTF_HOME/selftf/tf_job/classification/disML_new_api.py \
--ML_model=SVM \
--data_dir=$DATASET_BASE_PATH/kddb_splits \
--num_Features=29890095

for i in 1 2 3 4 5
do
  sleep 60
  if [[ $((i % 2)) -eq 0 ]];
   then sh bin/change_cluster_num.sh 0 $i;
   else sh bin/change_cluster_num.sh 0 -$i;
  fi
done