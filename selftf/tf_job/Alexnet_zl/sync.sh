scp -r bin selftf requirements.txt root@b1g37:pstuner/ && ssh root@b1g37 pstuner/bin/sync_code_slaves.sh && ssh root@b1g37 sh pstuner/bin/run_cnn_imagenet_single_config.sh

