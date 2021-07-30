import csv
import json
import logging
import os
import re
import subprocess
import sys
import time

from selftf.lib.common import PSTunerConfiguration, BaselineRecord

selftf_home = os.getenv("SELFTF_HOME")
if selftf_home is None:
    logging.error("env SELFTF_HOME is not set")
    sys.exit(-1)
base_path = "{}/log".format(selftf_home)
date_range = ('0718_021946', '1231_172935')

regex_folder_name = re.compile("^(.*)_(.*_.*)$") # example: SVM_0417_073032
regex_config_output = re.compile("Dry Run: learning_rate: (.*) ,batch_size: (.*) ,optimizer: (.*) ,number_worker: (.*) ,number_ps: (.*) ,num_intra_thread: (.*) ,num_inter_thread: (.*)")
regex_config_output2 = re.compile("Dry Run: (.*)$")

temp_folder = "/tmp/pstuner_log_%s" % str(time.time())
os.mkdir(temp_folder)
monitor_file = "monitor.out"
summary_csv = "summary.csv"
result_csv = os.path.join(temp_folder, "result.csv")

#  list all files
stdout = subprocess.check_output(["ls",base_path])
all_folder_list = stdout.split("\n")

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

        subprocess.check_output(["cp", "-r", os.path.join(base_path,folder_name), temp_folder])

        job_id = folder_name
        m = regex_folder_name.match(folder_name)
        model_name = m.group(1)

        local_folder_name = os.path.join(base_path, folder_name)
    # for each log folder
    #   grep the monitor.out | "Dry Run: "
    #   grep the configuration
        monitor_output = subprocess.check_output(["grep", "Dry Run:", os.path.join(local_folder_name, monitor_file)])
        m = regex_config_output.search(monitor_output)
        pstuner_config = None
        if m is not None:
            learning_rate = float(m.group(1))
            batch_size = int(m.group(2))
            optimizer = str(m.group(3))
            number_worker = int(m.group(4))
            number_ps = int(m.group(5))
            num_intra_thread = int(m.group(6))
        else:
            learning_rate = 0
            batch_size = 0
            optimizer = 0
            number_worker = 0
            number_ps = 0
            num_intra_thread = 0
            json_pstuner_config = regex_config_output2.search(monitor_output).group(1)
            dict_pstuner_config = eval(json_pstuner_config)
            pstuner_config = PSTunerConfiguration(py_dict=dict_pstuner_config)

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
            first_loss=first_loss,
            pstuner_config=pstuner_config
        )
        ret.append(record)
    except:
        subprocess.check_output(
            ["rm", "-fr", os.path.join(temp_folder, folder_name)])
        logging.exception("skip job log: %s" % folder_name)
#
# with open(result_csv, 'w+') as f:
#     w = csv.DictWriter(f, ["N_iterations", "N_workers", "N_intra", "Optimizer", "Learning_rate", "Batch_size", "n_partitions", "current_loss", "time_cost", "model_name", "job_id", "first_loss"])
#     w.writeheader()
#     w.writerows(map(lambda x:x.__dict__, ret))

print(temp_folder)

