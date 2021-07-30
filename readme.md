# Manual for MLTuner
## Installation

### Install Python Environment

```bash
chmod +x import.sh && \
./import.sh <hostname_of_master_node> <path_to_current_repo> <path_to_anaconda> && \
conda activate tftuner
```

### Install RabbitMQ

```bash
yum install -y erlang
wget https://github.com/rabbitmq/rabbitmq-server/releases/download/rabbitmq_v3_3_5/rabbitmq-server-3.3.5-1.noarch.rpm
yum install -y rabbitmq-server-3.3.5-1.noarch.rpm
echo [{rabbit, [{loopback_users, []}]}]. > /etc/rabbitmq/rabbitmq.config
nohup rabbitmq-server start & # Start rabbitmq server in the background or run `rabbitmq-server start` in a tmux screen
rabbitmqctl status # Check for RabbitMQ status
```

### Install MySQL

```bash
wget https://dev.mysql.com/get/mysql80-community-release-el7-3.noarch.rpm # Download and install MySQL
sudo rpm -Uvh mysql80-community-release-el7-3.noarch.rpm
sudo yum install mysql-community-server
sudo systemctl start mysqld.service
sudo systemctl status mysqld.service # Check for MySQL status

grep 'temporary password' /var/log/mysqld.log # Check for MySQL temporary password
mysql -uroot -p # Login MySQL
```

Within the MySQL shell:

```mysql
set global validate_password.policy=0; # Execute this command if you want to set a simple password
set global validate_password.length=1; # Execute this command if you want to set a password shorter than 8 characters
ALTER USER 'root'@'localhost' IDENTIFIED BY '<new_password>'; # Replace <new_password> with your own password
create database <dbname>; # Replace <dbname> with your own database name
exit
```

```bash
mysql -u root -p <dbname> < mltuner_db_structure.sql # Replace <dbname> with your own database name
```

## Usage

1. Add hostname of workers to file `slaves`. For example, 

    ```bash
    worker0
    worker1
    worker2
    ```

2. Modify the program to be tuned. Take `selftf/tf_job/sample/train_without_mltuner.py` as an example. Add the following four snippets to it and thus we get a training program compatible with MLTuner, `selftf/tf_job/sample/train.py`.

    1. Import MLTuner.

        ```python
        from selftf.lib.mltuner.mltuner_util import MLTunerUtil
        from selftf.lib.mltuner.mltuner_util import convert_model
		mltunerUtil = MLTunerUtil()
        ```
        
     2. Get/Set tuned parameters.
    
        ```python
        # Get non-hardware parameter
        optimizer = mltunerUtil.get_optimizer()
        num_worker = mltunerUtil.get_num_worker()
        worker_index = mltunerUtil.get_worker_index()
        # Set hardware parameter (Tensorflow version)
        session_config = mltunerUtil.get_tf_session_config()
        config = tf.estimator.RunConfig(train_distribute=strategy, session_config=session_config)
        # Set hardware parameter (Pytorch version)
        mltunerUtil.set_pytorch_hardware_param()
        # Set hardware parameter (Mxnet version)
        mltunerUtil.set_mxnet_hardware_param()
        ```
    
    3. Convert model for comparison right after the model is compiled.
    
        ```python
        FLAGS = tf.compat.v1.app.flags.FLAGS
        if FLAGS.get_model:
            convert_model(model, FLAGS.script_path)
            exit(0)
        ```
    
    4. Report program performance each step or at least in the last step.
    
        ```python
        mltunerUtil.report_iter_loss(step, loss, elapsed_time_in_ms)
        ```
    
3. Submit a job. Please make sure environment variable `SELFTF_HOME` mentioned in installation is set before launching.

    ```bash
    source bin/common.sh && \
    source bin/sync_code_slaves.sh && \
    sh bin/start_monitor.sh && \
    sh bin/start_all_agent.sh && \
    python selftf/client.py --action submit_job \
    --script python selftf/tf_job/sample/train.py --mode mltuner \
	--mysql_account <account> --mysql_password <password> --mysql_dbname <dbname>
    ```
    

## Important log files

1. `monitor.out`
    - Main monitor output
2. `log/{job id}`
    - Summary and monitor output backup
3. `/tmp/tftuner_agent.out`
    - Worker program output
