source $(dirname "$0")/common.sh
DATA_DIR="hdfs://ssd02:8020/user/root/train_data/imagenet-data"

cd $SELFTF_HOME
nohup python ${SELFTF_HOME}/selftf/tf_job/inception/inception/imagenet_distributed_train.py \
--batch_size=32 \
--data_dir=${DATA_DIR} \
--job_name='ps' \
--task_id=0 \
--ps_hosts='ssd02:2222' \
--worker_hosts='ssd07:2222,ssd08:2222' \
&

ssh root@ssd07 sh pstuner/bin/run_inception_worker.sh 0

ssh root@ssd08 sh pstuner/bin/run_inception_worker.sh 1
