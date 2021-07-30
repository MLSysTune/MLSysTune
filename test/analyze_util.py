import csv
import re
import subprocess
import sys

from selftf.lib import common
from selftf.lib.common import PSTunerConfiguration, BaselineRecord

overrun_runtime = 36000

regex_config_output = re.compile("Dry Run: learning_rate: (.*) ,batch_size: (.*) ,optimizer: (.*) ,number_worker: (.*) ,number_ps: (.*) ,num_intra_thread: (.*) ,num_inter_thread: (.*)")
regex_config_output2 = re.compile("Dry Run: (.*)$")
regex_we_got = re.compile("We got a new config: (.*)$")

regex_folder_name = re.compile("^(.*)_(.*_.*)$")  # example: SVM_0417_073032

class PSTunerConfigurationUtil():

    do_common_subexpression_elimination = 1,
    max_folded_constant_in_bytes = 1,
    do_function_inlining = 1,
    global_jit_level = 0,
    infer_shapes = 1,
    enable_bfloat16_sendrecv = 0,
    place_pruned_graph = 0,
    json_dict = None

def get_model_name_by_job_id(job_id):
    m = regex_folder_name.search(job_id)
    return m.group(1)

def get_baseline_record_bymonitor_out_dry_run(monitor_file, summary_csv, loss_data_path, job_id):
    monitor_output = subprocess.check_output(
        ["grep", "Dry Run:", monitor_file])
    m = regex_config_output.search(monitor_output)
    pstuner_config = None
    if m is not None:
        # Eventually we remove this block
        learning_rate = float(m.group(1))
        batch_size = int(m.group(2))
        optimizer = str(m.group(3))
        number_worker = int(m.group(4))
        number_ps = int(m.group(5))
        num_intra_thread = int(m.group(6))
        pstuner_config = PSTunerConfiguration(
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=common.get_optimizer_by_name(optimizer),
            num_worker=number_worker,
            num_ps=number_ps,
            intra_op_parallelism_threads=num_intra_thread,
            inter_op_parallelism_threads=16 - num_intra_thread,
        )
    else:
        learning_rate = 0
        batch_size = 0
        optimizer = 0
        number_worker = 0
        number_ps = 0
        num_intra_thread = 0
        json_pstuner_config = regex_config_output2.search(monitor_output).group(
            1)
        dict_pstuner_config = eval(json_pstuner_config)
        pstuner_config = PSTunerConfiguration(py_dict=dict_pstuner_config)

    #   grep the summary.csv
    with open(summary_csv) as summary_csv_file:
        reader = csv.DictReader(summary_csv_file)
        result = reader.next()
        runtime = float(result["runtime_sec"])
        try:
            final_loss = result["final_lossos_size"]  # Typo
        except:
            final_loss = result["final_loss"]
        total_iteration = int(result["total_iteration"])
        target_loss = float(result["target_loss"])

    with open(loss_data_path) as loss_data_file:
        reader = csv.DictReader(loss_data_file)
        result = reader.next()
        first_loss = float(result["loss"])

    if runtime == 0:
        if float(final_loss) > float(first_loss):
            # Diverge case
            runtime = sys.float_info.max
        else:
            # Converge but not finished
            # we estimate here
            # if
            target_loss_dif = float(first_loss) - float(target_loss)
            current_loss_dif = float(first_loss) - float(final_loss)
            runtime = (target_loss_dif * overrun_runtime) / current_loss_dif

            # the runtime is worse..... make it diverge
            if runtime > overrun_runtime * 1000:
                runtime = sys.float_info.max
                final_loss = first_loss +100

    record = BaselineRecord(
        N_iterations=total_iteration,
        N_workers=number_worker,
        N_intra=num_intra_thread,
        Optimizer=optimizer,
        Learning_rate=learning_rate,
        Batch_size=batch_size,
        n_partitions=108,
        current_loss=final_loss,
        time_cost=runtime,
        job_id=job_id,
        model_name=get_model_name_by_job_id(job_id),
        first_loss=first_loss,
        pstuner_config=pstuner_config
    )

    return record


def get_summary_dict(path):
    with open(path) as summary_csv_file:
        reader = csv.DictReader(summary_csv_file)
        result = reader.next()
    return result
