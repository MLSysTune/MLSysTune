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

class BaselineRecord(object):
    def __init__(self,
        N_iterations,
        N_workers,
        N_intra,
        Optimizer,
        Learning_rate,
        Batch_size,
        n_partitions,
        current_loss,
        time_cost,
        model_name,
        job_id,
        first_loss
    ):
        self.N_iterations = N_iterations
        self.N_workers = N_workers
        self.N_intra = N_intra
        self.Optimizer = Optimizer
        self.Learning_rate = Learning_rate
        self.Batch_size = Batch_size
        self.n_partitions = n_partitions
        self.current_loss = current_loss
        self.time_cost = time_cost
        self.model_name = model_name
        self.job_id = job_id
        self.first_loss = first_loss


selftf_home = os.getenv("SELFTF_HOME")
if selftf_home is None:
    logging.error("env SELFTF_HOME is not set")
    sys.exit(-1)
base_path = "{}/log".format(selftf_home)
master_host = "b1g37"
master_user = "root"
master_password = "dbg_123"
master_private_key = "/Users/cyliu/.ssh/id_rsa"
date_range = ('0623_110531', '0710_172935')

regex_folder_name = re.compile("^(.*)_(.*_.*)$") # example: SVM_0417_073032
regex_config_output = re.compile("Dry Run: learning_rate: (.*) ,batch_size: (.*) ,optimizer: (.*) ,number_worker: (.*) ,number_ps: (.*) ,num_intra_thread: (.*) ,num_inter_thread: (.*)")

temp_folder = "/tmp/pstuner_log_%s" % str(time.time())
os.mkdir(temp_folder)
monitor_file = "monitor.out"
summary_csv = "summary.csv"
result_csv = os.path.join(temp_folder, "result.csv")

def get_ssh_connection():
    # ssh list all files
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(WarningPolicy())
    ssh.connect(master_host, username=master_user, password=master_password)
    return ssh

ssh = get_ssh_connection()
stdin, stdout, stderr = ssh.exec_command("ls %s" % base_path)
all_folder_list = stdout.read().split("\n")
ssh.close()

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

# download the log file
for folder_name in target_folder_list:
    try:
        ssh = get_ssh_connection()
        scp = SCPClient(ssh.get_transport())

        job_id = folder_name
        m = regex_folder_name.match(folder_name)
        model_name = m.group(1)

        scp.get(os.path.join(base_path, folder_name), temp_folder, recursive=True)

        local_folder_name = os.path.join(temp_folder, folder_name)
    # for each log folder
    #   grep the monitor.out | "Dry Run: "
    #   grep the configuration
        monitor_output = subprocess.check_output(["grep", "Dry Run:", os.path.join(local_folder_name, monitor_file)])
        m = regex_config_output.search(monitor_output)
        learning_rate = float(m.group(1))
        batch_size = int(m.group(2))
        optimizer = str(m.group(3))
        number_worker = int(m.group(4))
        number_ps = int(m.group(5))
        num_intra_thread = int(m.group(6))

    #   grep the summary.csv
        with open(os.path.join(local_folder_name, summary_csv)) as summary_csv_file:
            reader = csv.DictReader(summary_csv_file)
            result = reader.next()
            runtime = result["runtime_sec"]
            try:
                final_loss = result["final_lossos_size"] # Typo
            except:
                final_loss = result["final_loss"]
            total_iteration = result["total_iteration"]
        with open(os.path.join(local_folder_name, job_id+"iteration_training_data.csv")) as loss_data_file:
            reader = csv.DictReader(loss_data_file)
            result = reader.next()
            first_loss = result["loss"]

        record = BaselineRecord(
            N_iterations = total_iteration,
            N_workers = number_worker,
            N_intra = num_intra_thread,
            Optimizer = optimizer,
            Learning_rate = learning_rate,
            Batch_size = batch_size,
            n_partitions = 108,
            current_loss = final_loss,
            time_cost = runtime,
            job_id=folder_name,
            model_name=model_name,
            first_loss=first_loss
        )
        ret.append(record)
        ssh.close()
    except:
        logging.exception("skip job log: %s" % folder_name)

with open(result_csv, 'w+') as f:
    w = csv.DictWriter(f, ["N_iterations", "N_workers", "N_intra", "Optimizer", "Learning_rate", "Batch_size", "n_partitions", "current_loss", "time_cost", "model_name", "job_id", "first_loss"])
    w.writeheader()
    w.writerows(list(map(lambda x:x.__dict__, ret)))

ssh.close()
print("Check result in: %s" % result_csv)

