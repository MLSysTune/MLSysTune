import argparse
import json

import os

KEY_AMQP_HOST = "amqp_host"
KEY_AMQP_PORT = "amqp_port"
KEY_AMQP_USER = "amqp_user"
KEY_AMQP_PASSWORD = "amqp_password"
KEY_HOSTNAME = "hostname"

KEY_AGENT_WORKSPACE = "agent_workspace_path"
KEY_TFTUNER_HOME = "tftuner_home"
KEY_AGENT_TFPORT = "agent_tfport"
KEY_AGENT_SSH_KEY = "agent_ssh_key"
KEY_AGENT_NUMBER_OF_THREADS = "agent_num_thread"

KEY_RECONFIG_NUM_ITER_EACH_OS = "num_iter_per_os"
KEY_RECONFIG_OS_SIZE = "os_size"
KEY_ONLINE_RECONFIG_OS_SIZE = "online_reconfig_size"
KEY_ONLINE_RECONFIG_NUM_ITER_PER_OS = "online_num_iter_per_os"
KEY_ESTIMATION_FUNC = "estimation_func"

KEY_TARGET_LOSS = "target_loss"
KEY_ML_MODEL = "ml_model"
KEY_BATCH_SIZE = "batch_size"
KEY_LEARNING_RATE = "learning_rate"
KEY_OPTIMIZER = "optimizer"
KEY_NUM_WORKER = "num_worker"
KEY_NUM_INTRA_THREAD = "num_intra_thread"
KEY_n_partition = "n_partition"

KEY_LEARNING_RATE_RANGE = "learning_rate_range"
KEY_BATCH_SIZE_RANGE = "batch_size_range"

# # For Pure TF dry run
KEY_MODE = "mode"
KEY_JSON_DICT_PSTUNER_CONFIG = "json_dict_pstuner_config"

MODE_VALUE_DRY_RUN="dry_run"
MODE_VALUE_SELFTF = "selftf"
MODE_VALUE_MLTUNER = "mltuner"
MODE_VALUE_SELFTF_RECONFIG_ZERO = "selftf_reconfig_0"
MODE_VALUE_TEST_RECONFIGSCHEME_ONE = "test_reconfig_1"
MODE_VALUE_TEST_RECONFIGSCHEME_ZERO = "test_reconfig_0"
MODE_VALUE_TEST_RECONFIGSCHEME_TWO = "test_reconfig_2"
MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME="test_estimation_func_remaining_time"
MODE_VALUES = [
    MODE_VALUE_DRY_RUN,
    MODE_VALUE_SELFTF,
    MODE_VALUE_SELFTF_RECONFIG_ZERO,
    MODE_VALUE_MLTUNER,
    MODE_VALUE_TEST_RECONFIGSCHEME_ONE,
    MODE_VALUE_TEST_RECONFIGSCHEME_ZERO,
    MODE_VALUE_TEST_RECONFIGSCHEME_TWO,
    MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME,
]

# Key for estimation function
KEY_TEST_ESTIMATION_FUNC_CONFIG_IDX = MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME + "_STOP_IDX_CONFIG"

KEY_MYSQL_ACCOUNT = "mysql_account"
KEY_MYSQL_PASSWORD = "mysql_password"
KEY_MYSQL_DBNAME = "mysql_dbname"

conf_parser = argparse.ArgumentParser()

conf_parser.add_argument("--" + KEY_AMQP_HOST, default="localhost")
conf_parser.add_argument("--" + KEY_AMQP_PORT, default="5672")
conf_parser.add_argument("--" + KEY_AMQP_USER, default="guest")
conf_parser.add_argument("--" + KEY_AMQP_PASSWORD, default="guest")
conf_parser.add_argument("--" + KEY_HOSTNAME, default="localhost")
conf_parser.add_argument("--" + KEY_AGENT_WORKSPACE, default=".")
conf_parser.add_argument("--" + KEY_TFTUNER_HOME, default=".")
conf_parser.add_argument("--" + KEY_AGENT_TFPORT, default=2222)
conf_parser.add_argument("--" + KEY_AGENT_SSH_KEY, default="id_rsa")
conf_parser.add_argument("--" + KEY_AGENT_NUMBER_OF_THREADS, default=16)

conf_parser.add_argument("--" + KEY_RECONFIG_NUM_ITER_EACH_OS, default=5)
conf_parser.add_argument("--" + KEY_RECONFIG_OS_SIZE, default=20)
conf_parser.add_argument("--" + KEY_ONLINE_RECONFIG_OS_SIZE, default=100)
conf_parser.add_argument("--" + KEY_ONLINE_RECONFIG_NUM_ITER_PER_OS, default=200)

conf_parser.add_argument("--" + KEY_TARGET_LOSS, default=0.5)
conf_parser.add_argument("--" + KEY_ML_MODEL, default="ML Model")
conf_parser.add_argument("--" + KEY_BATCH_SIZE, default=1000)
conf_parser.add_argument("--" + KEY_LEARNING_RATE, default=0.001)

conf_parser.add_argument("--" + KEY_ESTIMATION_FUNC, default="bo")

# Not fully implemented
conf_parser.add_argument("--" + KEY_LEARNING_RATE_RANGE, default="[0.00001,0.0001]")
conf_parser.add_argument("--" + KEY_BATCH_SIZE_RANGE, default="[1000,5000]")

conf_parser.add_argument("--" + KEY_MODE, default=MODE_VALUE_SELFTF)
conf_parser.add_argument("--" + KEY_OPTIMIZER, default="Adam")
conf_parser.add_argument("--" + KEY_NUM_WORKER, default=35)
conf_parser.add_argument("--" + KEY_NUM_INTRA_THREAD, default=15)
conf_parser.add_argument("--" + KEY_n_partition, default=1)

# for MODE_VALUE_TEST_ESTIMATION_FUNC_REMAINING_TIME
conf_parser.add_argument("--" + KEY_TEST_ESTIMATION_FUNC_CONFIG_IDX, default=0)
conf_parser.add_argument("--" + KEY_JSON_DICT_PSTUNER_CONFIG , default="")

conf_parser.add_argument("--" + KEY_MYSQL_ACCOUNT , default="root")
conf_parser.add_argument("--" + KEY_MYSQL_PASSWORD , default="12345678")
conf_parser.add_argument("--" + KEY_MYSQL_DBNAME , default="tftuner")

default_config = None


class Config(object):
    def __init__(self, dict, remain):
        self.dict = dict
        self.remain = remain

    def get_amqp_host(self):
        return self.dict[KEY_AMQP_HOST]

    def get_amqp_port(self):
        return self.dict[KEY_AMQP_PORT]

    def get_amqp_user(self):
        return self.dict[KEY_AMQP_USER]

    def get_amqp_password(self):
        return self.dict[KEY_AMQP_PASSWORD]

    def get_hostname(self):
        return self.dict[KEY_HOSTNAME]

    def get_agent_workspace(self):
        return self.dict[KEY_AGENT_WORKSPACE]

    def get_agent_tfport(self):
        return self.dict[KEY_AGENT_TFPORT]

    def get(self, key):
        return self.dict[key]

    def get_remain(self):
        """
        :rtype: list[string]
        """
        return self.remain

    def get_target_loss(self):
        return self.dict[KEY_TARGET_LOSS]

    def get_tftuner_home(self):
        return self.dict[KEY_TFTUNER_HOME]

    def get_ml_model(self):
        return self.dict[KEY_ML_MODEL]

    def get_batch_size(self):
        return self.dict[KEY_BATCH_SIZE]

    def get_learning_rate(self):
        return self.dict[KEY_LEARNING_RATE]

    def get_log_dir(self):
        return os.path.join(self.get_tftuner_home(), "log")

    def get_repo_dir(self):
        return os.path.join(self.get_tftuner_home(), "repo")

    def get_job_log_dir(self, job_id ):
        return os.path.join(self.get_log_dir(), job_id)

    def get_online_os_size(self):
        return self.dict[KEY_ONLINE_RECONFIG_OS_SIZE]

    def get_learning_rate_range(self):
        """
        :return:
        :rtype: (float,float)
        """
        values = json.loads(self.dict[KEY_LEARNING_RATE_RANGE])
        return float(values[0]), float(values[1])

    def get_batch_size_range(self):
        """
        :return:
        :rtype: (int, int)
        """
        values = json.loads(self.dict[KEY_BATCH_SIZE_RANGE])
        return int(values[0]), int(values[1])
    
    def get_mysql_account(self):
        return self.dict[KEY_MYSQL_ACCOUNT]
    
    def get_mysql_password(self):
        return self.dict[KEY_MYSQL_PASSWORD]
    
    def get_mysql_dbname(self):
        return self.dict[KEY_MYSQL_DBNAME]


def get_config():
    args, unknown_args = conf_parser.parse_known_args()
    default_config = Config(vars(args), unknown_args)
    return default_config


optimizer_list = ["Adam","SGD","Adadelta","RMSProp","Momentum","RMSProp_Imagenet"]