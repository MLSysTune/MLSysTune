export DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

export SELFTF_MASTER_NODE="ssd33"
export AMPQ_MASTER_NODE=$SELFTF_MASTER_NODE
export HDFS_MASTER_NODE=$SELFTF_MASTER_NODE

export SELFTF_HDFS_HOME="hdfs://$HDFS_MASTER_NODE:8020/user/root/pstuner"
export DATASET_BASE_PATH="hdfs://$HDFS_MASTER_NODE:8020/user/root/train_data"

export PYTHONPATH=$PYTHONPATH:$SELFTF_HOME
export SELFTF_NUM_COMPUTE_NODE=$(wc -l < $SELFTF_HOME/slaves)
export SELFTF_NUM_THREAD=$(grep -c ^processor /proc/cpuinfo)

export SELFTF_FILE_AGENT_PID="/tmp/tftuner_agent.pid"
export SELFTF_FILE_AGENT_OUT="/tmp/tftuner_agent.out"

export SELFTF_CONDA_ENV="tftuner"

export SELFTF_MAIN_NIC="em1"
ulimit -n 65535
