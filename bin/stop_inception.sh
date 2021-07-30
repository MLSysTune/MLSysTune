

ps -ef | grep python| grep 'imagenet_distributed_train.py' | awk {'print $2'} | xargs kill

ssh root@ssd07 \
ps -ef | grep python| grep 'imagenet_distributed_train.py' | awk {'print $2'} | xargs kill


ssh root@ssd08 \
ps -ef | grep python| grep 'imagenet_distributed_train.py' | awk {'print $2'} | xargs kill
