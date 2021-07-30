source $(dirname "$0")/common.sh

export PYTHONPATH=$PYTHONPATH:$SELFTF_HOME
DATA_DIR=hdfs://ssd02:8020/user/root/train_data/imagenet-data
cd $SELFTF_HOME
nohup python selftf/tf_job/inception/inception/imagenet_distributed_train.py \
--batch_size=32 \
--data_dir=${DATA_DIR} \
--job_name='worker' \
--task_id=$1 \
--ps_hosts='ssd02:2222' \
--worker_hosts='ssd07:2222,ssd08:2222' \
&> run_inception.log &
