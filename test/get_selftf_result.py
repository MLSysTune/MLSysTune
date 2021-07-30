import csv
import logging
import os
import re
import subprocess
import sys
import time

import paramiko
from paramiko import WarningPolicy
from scp import SCPClient
import selftf.lib.common

selftf_home = os.getenv("SELFTF_HOME")
if selftf_home is None:
    logging.error("env SELFTF_HOME is not set")
    sys.exit(-1)
base_path = "{}/log".format(selftf_home)
master_host = "b1g37"
master_user = "root"
master_password = "dbg_123"
master_private_key = "/Users/cyliu/.ssh/id_rsa"
date_range = ('0621_143324', '0621_220215')

regex_folder_name = re.compile("^(.*)_(.*_.*)$") # example: SVM_0417_073032
regex_config_output = re.compile("Dry Run: learning_rate: (.*) ,batch_size: (.*) ,optimizer: (.*) ,number_worker: (.*) ,number_ps: (.*) ,num_intra_thread: (.*) ,num_inter_thread: (.*)")

temp_folder = "/tmp/pstuner_log_%s" % str(time.time())
os.mkdir(temp_folder)
monitor_file = "monitor.out"
summary_csv = "summary.csv"
result_csv = os.path.join(temp_folder, "result.csv")

# ssh list all files
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(WarningPolicy())
ssh.connect(master_host, username=master_user, password=master_password)
stdin, stdout, stderr = ssh.exec_command("ls %s" % base_path)
all_folder_list = stdout.read().split("\n")

# filter all file base on 'date_range'
def folder_filter(file_name):
    m = regex_folder_name.match(file_name)
    if m is None:
        return False
    date = m.group(2)
    if date >= date_range[0] and date <= date_range[1]:
        return True
    else:
        return False
target_folder_list = filter(folder_filter, all_folder_list)

ret = []

def is_selftf(monitor_file_path):
    subprocess.check_output(
        ["grep", "Best training conf", monitor_file_path])
    return True

# download the log file
scp = SCPClient(ssh.get_transport())
for folder_name in target_folder_list:
    try:

        job_id = folder_name
        m = regex_folder_name.match(folder_name)
        model_name = m.group(1)

        scp.get(os.path.join(base_path, folder_name), temp_folder, recursive=True)

        local_folder_name = os.path.join(temp_folder, folder_name)

        if not is_selftf(os.path.join(local_folder_name,monitor_file)):
            continue

        #   grep the summary.csv
        with open(os.path.join(local_folder_name,
                               summary_csv)) as summary_csv_file:
            reader = csv.DictReader(summary_csv_file)
            result = reader.next()
            record = selftf.lib.common.Summary()
            record.__dict__ = result

            ret.append(record)
    except:
        logging.exception("skip job log: %s" % folder_name)

with open(result_csv, 'w+') as f:
    w = csv.DictWriter(f, ["job_id",
                           "runtime_sec",
                           "avg_recovery_time_sec",
                           "target_loss",
                           "ml_model",
                           "batch_size",
                           "learning_rate",
                           "final_loss",
                           "os_size",
                           "n_iter_per_os",
                           "online_os_size",
                           "online_n_iter_per_os",
                           "check_reconfig",
                           "do_reconfig",
                           "total_iteration",
                           "init_duration",
                           "estimation_func"])
    w.writeheader()
    w.writerows(list(map(lambda x: x.__dict__, ret)))


ssh.close()
logging.info("Check result in: %s" % result_csv)

