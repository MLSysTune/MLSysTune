import csv
import json
import logging

import os
import shutil

import numpy
import pandas
import re
import subprocess

from selftf.lib import common
from selftf.lib.common import BaselineRecord
from test.analyze_util import regex_folder_name, get_model_name_by_job_id, \
    get_baseline_record_bymonitor_out_dry_run, regex_we_got, overrun_runtime, \
    get_summary_dict
from test.paper_script import script_error_rank

# workloads = ["ALEXNET_IMAGENET"]
workloads = ["LR","SVM_BIG","CNN","ALEXNET_IMAGENET"]

gnu_plot_base_path = "/Users/cyliu/Documents/selftf_paper/gnuplot_script"
gnu_plot_completition_time_path = os.path.join(gnu_plot_base_path,"completion_time")
loss_graph_path = os.path.join(gnu_plot_base_path,"loss_time")
paper_base_path = "/Users/cyliu/Documents/selftf_paper"
paper_image_base_path = os.path.join(paper_base_path, "images")
paper_table_base_path = os.path.join(paper_base_path, "tables")

dict_completition_time_target_file = {
    "CNN":"nntime.dat",
    "LR":"lrtime.dat",
    "SVM_BIG":"svmtime.dat",
    "ALEXNET_IMAGENET":"imagenettime.dat"
}

dict_workload_worst_job_id = {}
base_path = "/Users/cyliu/Documents/SampleData"

overtime_mock_time = overrun_runtime

def get_file_path(job_type, job_id, file):
    if file == "iteration_training_data.csv":
        file = job_id+"iteration_training_data.csv"
    model_name = regex_folder_name.match(job_id).group(1)
    return os.path.join(base_path, job_type, model_name, job_id, file)

def get_completiton_time_gnu_data(baseline_dataframe, selftf_dataframe):
    """
    :param pandas.DataFrame baseline_dataframe:
    :param pandas.DataFrame selftf_dataframe:
    :return:
    """
    target_baseline_worst_job_id, target_baseline_average_job_id, \
    target_selftf_job_id, target_baseline_best_job_id = get_ids_by_dataframe(selftf_dataframe, baseline_dataframe)

    filtered_baseline_tf = df_baseline.loc[
        df_baseline["current_loss"] < df_baseline["first_loss"]][
        ["job_id", "time_cost"]]

    return {
        "worst": baseline_dataframe.loc[baseline_dataframe["job_id"] == target_baseline_worst_job_id]["time_cost"].values[0],
        "average": filtered_baseline_tf["time_cost"].mean(),
        "optimal": baseline_dataframe.loc[baseline_dataframe["job_id"] == target_baseline_best_job_id]["time_cost"].values[0],
        "selftf": selftf_dataframe.loc[selftf_dataframe["job_id"] == target_selftf_job_id]["time_cost"].values[0]
    }


def write_completion_time_gnu_plot(worst, average, optimal, selftf, path):
    """
    in format
    d	Worst	Average	SelfTF	Optimal
    0	12328	0	0	0
    1	0	2888	0	0
    2	0	0	1309	0
    3	0	0	0	2076
    :param worst:
    :param average:
    :param optimal:
    :param selftf:
    :return:
    """
    ret_dict = [
        {
            "d":0,
            "Worst":worst,
            "Average":0,
            "SelfTF":0,
            "Optimal":0
        },{
            "d":1,
            "Worst":0,
            "Average":average,
            "SelfTF":0,
            "Optimal":0
        },{
            "d":2,
            "Worst":0,
            "Average":0,
            "SelfTF":selftf,
            "Optimal":0
        },{
            "d":3,
            "Worst":0,
            "Average":0,
            "SelfTF":0,
            "Optimal":optimal
        }
    ]
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["d", "Worst","Average","SelfTF","Optimal"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(ret_dict)

def get_first_iteration_group_by_config_idx(training_data_dataframe):
    """
    step	local_step	timestamp	elapsed_time_sec	loss	config_idx	ps_num	worker_num	intra_op_parallelism_threads	inter_op_parallelism_threads	optimizer	learning_rate	batch_size
        to
    config_idx, iteration number, the time_zero  and loss
    :param pandas.DataFrame training_data_dataframe:
    :return:
    """
    return training_data_dataframe.loc[training_data_dataframe.groupby(["config_idx"])["timestamp"].idxmin()].loc[:,["config_idx","timestamp_zero","loss","step"]]


def get_self_loss_graph_df(monitor_path, training_data_dataframe):
    """
    :param monitor_path:
    :param pandas.DataFrame training_data_dataframe:
    :return:
    """
    # initialization phase -- LHS config
    # Since we use pop() reverse the result
    list_config = []
    regex_lhs_conf = re.compile("Generated LHS conf list: (.*)$")
    lhs_line = subprocess.check_output(["grep", "Generated LHS conf list:", monitor_path])
    assert lhs_line
    m = regex_lhs_conf.search(lhs_line)

    if m is not None:
        list_lhs_config_dict = eval(m.group(1))
        assert isinstance(list_lhs_config_dict, list)
        list_lhs_config_dict.reverse()
        list_config.extend(list_lhs_config_dict)
        # Since we let selftuner run with original config immediately when entering
        # online tuning phase
        list_config.append(list_lhs_config_dict[-1].copy())
    else:
        raise Exception()

    config_idx_to_remove = len(list_lhs_config_dict)-1
    first_online_tuning_config_idx = len(list_lhs_config_dict)

    # Online tuning phase grep "WE got"

    we_got_lines = subprocess.check_output(["grep", "We got a new config", monitor_path]).split("\n")
    for line in we_got_lines:
        if line == "":
            continue
        m = regex_we_got.search(line)
        if m is not None:
            config = eval(m.group(1))
            list_config.append(config)
        else:
            raise Exception()

    # add index
    for idx, config in enumerate(list_config):
        config["idx"] = idx

    df_config_idx_first_iteration = get_first_iteration_group_by_config_idx(training_data_dataframe)
    df_list_config = pandas.DataFrame(list_config)
    ret = df_list_config.merge(df_config_idx_first_iteration, how="left", left_on="idx", right_on="config_idx").drop("idx",axis=1)

    # filter out the unwanted one
    ret = ret.loc[ret["config_idx"]!=config_idx_to_remove]
    ret_first_config_online_tuning = ret.loc[ret["config_idx"] == first_online_tuning_config_idx]
    return ret, ret_first_config_online_tuning

def get_dataframe_add_time_sec_to_training_data(training_data_path):
    "step,local_step,timestamp,elapsed_time_sec,loss,config_idx,ps_num,worker_num,intra_op_parallelism_threads,inter_op_parallelism_threads,optimizer,learning_rate,batch_size"
    df = pandas.DataFrame.from_csv(training_data_path, index_col=None)
    df["timestamp"] = df["timestamp"].astype(float)
    first_timestamp = df["timestamp"].min()
    df["timestamp_zero"] = df.eval("timestamp - "+str(first_timestamp))
    return df


def get_loss_graph(df_selftf, df_baseline):
    """
    :type df_selftf: pandas.DataFrame
    """
    target_baseline_worst_job_id, target_baseline_average_job_id, target_selftf_job_id, target_baseline_best_job_id = get_ids_by_dataframe(df_selftf, df_baseline)

    # Get the model name
    model_name = get_model_name_by_job_id(target_selftf_job_id)

    # Get the path of training data .csv
    selftf_loss_csv_path = get_file_path("selftf", job_id=target_selftf_job_id, file="iteration_training_data.csv")
    baseline_average_loss_csv_path = get_file_path("baseline", job_id=target_baseline_average_job_id, file="iteration_training_data.csv")
    baseline_best_loss_csv_path = get_file_path("baseline", job_id=target_baseline_best_job_id, file="iteration_training_data.csv")
    baseline_worst_loss_csv_path = get_file_path("baseline", job_id=target_baseline_worst_job_id, file="iteration_training_data.csv")

    # add timestamp data
    ret_df = list(map(get_dataframe_add_time_sec_to_training_data,
                 [selftf_loss_csv_path, baseline_average_loss_csv_path, baseline_best_loss_csv_path, baseline_worst_loss_csv_path]))

    # Get the monitor.out of selftf
    selftf_loss_monitor_path = os.path.join(base_path, "selftf",model_name,target_selftf_job_id,"monitor.out")
    config_timestamp_df, online_tuning_df = get_self_loss_graph_df(selftf_loss_monitor_path,
                                                                   ret_df[0])

    output_path = os.path.join(loss_graph_path, model_name)
    try:
        shutil.rmtree(output_path)
    except:
        pass
    os.mkdir(output_path)

    # subprocess.check_output(["cp",selftf_loss_csv_path, os.path.join(output_path,"selftf_loss_training_data.csv")])
    # subprocess.check_output(["cp",baseline_average_loss_csv_path, os.path.join(output_path,"baseline_average_loss_training_data.csv")])
    # subprocess.check_output(["cp",baseline_best_loss_csv_path, os.path.join(output_path,"baseline_best_loss_training_data.csv")])
    # subprocess.check_output(["cp",baseline_worst_loss_csv_path, os.path.join(output_path,"baseline_worst_loss_training_data.csv")])
    ret_df[0].to_csv(os.path.join(output_path,"selftf_loss_training_data.csv"))
    ret_df[1].to_csv(os.path.join(output_path,"baseline_average_loss_training_data.csv"))
    ret_df[2].to_csv(os.path.join(output_path,"baseline_best_loss_training_data.csv"))
    ret_df[3].to_csv(os.path.join(output_path,"baseline_worst_loss_training_data.csv"))

    config_timestamp_df.to_csv(os.path.join(output_path,"config.csv"))
    online_tuning_df.to_csv(os.path.join(output_path,"online_tuning_first.csv"))

def get_dataframe_baseline(path):
    ret = []
    stdout = subprocess.check_output(["ls", path])
    all_folder_list = stdout.split("\n")[:-1]
    for folder_name in all_folder_list:
        record = get_baseline_record_bymonitor_out_dry_run(
            monitor_file=os.path.join(path, folder_name, "monitor.out"),
            summary_csv=os.path.join(path, folder_name, "summary.csv"),
            loss_data_path=os.path.join(path, folder_name, folder_name+"iteration_training_data.csv"),
            job_id=folder_name
        )
        ret.append(record)

    # For completion time
    df = pandas.DataFrame(list(map(lambda a: a.__dict__, ret)))
    return df


def get_dataframe_selftf_or_reconfig0(path, skip_config=False):
    ret = []
    stdout = subprocess.check_output(["ls", path])
    all_folder_list = stdout.split("\n")
    for folder_name in all_folder_list:
        if folder_name == "":
            continue
        job_id = folder_name
        m = regex_folder_name.search(folder_name)
        model_name = m.group(1)

        local_folder_name = os.path.join(path, folder_name)
        monitor_path = os.path.join(local_folder_name, "monitor.out")

        learning_rate, batch_size, optimizer = common.get_default_learning_rate_batch_size_optimizer(
            model_name
        )

        # get the last config
        pstuner_config = None
        if not skip_config:
            line = subprocess.check_output(
                ["grep", "We got a new config", monitor_path]).split("\n")[-2]
            m = regex_we_got.search(line)
            if m is not None:
                pstuner_config_dict = eval(m.group(1))
                pstuner_config=common.PSTunerConfiguration(
                    py_dict=pstuner_config_dict)
            else:
                raise Exception()

        #   grep the summary.csv
        summary_path = os.path.join(local_folder_name, summary_csv)
        result = get_summary_dict(summary_path)
        runtime = result["runtime_sec"]
        try:
            final_loss = result["final_lossos_size"]  # Typo
        except:
            final_loss = result["final_loss"]
        total_iteration = result["total_iteration"]


        with open(os.path.join(local_folder_name,
                               job_id + "iteration_training_data.csv")) as loss_data_file:
            reader = csv.DictReader(loss_data_file)
            result = reader.next()
            first_loss = result["loss"]

        record = BaselineRecord(
            N_iterations=total_iteration,
            N_workers=0,
            N_intra=0,
            Optimizer=optimizer,
            Learning_rate=learning_rate,
            Batch_size=batch_size,
            n_partitions=108,
            current_loss=final_loss,
            time_cost=runtime,
            job_id=folder_name,
            model_name=model_name,
            first_loss=first_loss,
            pstuner_config=pstuner_config
        )
        ret.append(record)

    # For completion time
    df = pandas.DataFrame(list(map(lambda a: a.__dict__, ret)))
    return df

def get_ids_by_dataframe(df_selftf, df_baseline):
    target_selftf_job_id = df_selftf.loc[df_selftf["time_cost"].idxmin()][
        "job_id"]
    filtered_baseline_tf = df_baseline.loc[
        df_baseline["current_loss"] < df_baseline["first_loss"]][
        ["job_id", "time_cost"]]
    baseline_mean = filtered_baseline_tf.time_cost.mean()
    target_baseline_average_job_id = df_baseline.loc[
        (df_baseline.time_cost - baseline_mean).abs().argsort()[:1]][
        "job_id"].values[0]
    target_baseline_best_job_id = df_baseline.loc[df_baseline["time_cost"].idxmin()]["job_id"]
    v = filtered_baseline_tf["time_cost"].max()
    again = filtered_baseline_tf.loc[filtered_baseline_tf["time_cost"].idxmax()]
    target_baseline_worst_job_id = filtered_baseline_tf.loc[filtered_baseline_tf["time_cost"] == v]["job_id"].values[0]

    return target_baseline_worst_job_id, target_baseline_average_job_id, \
           target_selftf_job_id, target_baseline_best_job_id

def get_recovery_time_list(monitor_path):
    regex_recovery_list = re.compile("current recovery time map:(.*)$")
    line = subprocess.check_output(["grep", "current recovery time map:", monitor_path])
    m = regex_recovery_list.search(line.split("\n")[-2])
    if m is None:
        raise Exception()
    return eval(m.group(1))


def get_reconfig_table(df_selftf, df_selftf_reconfig0):
    """
    Row model
    Column: [Total runtime	reconfig overhead	%	average time] x2 (for reconfig0 and reconfig2)
    :param df_selftf:
    :param df_selftf_reconfig0:
    :return:
    """

    time_cost_reconfig0 = df_selftf_reconfig0["time_cost"].max()
    time_cost_reconfig2 = df_selftf["time_cost"].min()
    
    index_reconfig0 = df_selftf_reconfig0["time_cost"].idxmax()
    index_reconfig2 = df_selftf["time_cost"].idxmin()
    
    target_reconfig0_selftf_job_id = df_selftf_reconfig0.loc[index_reconfig0][
        "job_id"]
    target_selftf_job_id = df_selftf.loc[index_reconfig2][
        "job_id"]

    reconfig0_monitor_path = get_file_path("selftf_reconfig_0", target_reconfig0_selftf_job_id, "monitor.out")
    reconfig2_monitor_path = get_file_path("selftf", target_selftf_job_id, "monitor.out")

    reconfig0_recovery_list = get_recovery_time_list(reconfig0_monitor_path)[1:]
    reconfig2_recovery_list = get_recovery_time_list(reconfig2_monitor_path)[1:]
    
    reconfig0_total_recovery = sum(reconfig0_recovery_list)
    reconfig2_total_recovery = sum(reconfig2_recovery_list)

    # get sumamry file
    selftf_summary_path = get_file_path("selftf", target_selftf_job_id, "summary.csv")
    summary_dict = get_summary_dict(selftf_summary_path)

    return {
        "num_do_reconfig": int(summary_dict["do_reconfig"]) + int(summary_dict["os_size"]),
        "reconfig0_total_runtime":time_cost_reconfig0,
        "reconfig0_reconfig_overhead":reconfig0_total_recovery,
        "reconfig0_precentage":(reconfig0_total_recovery/time_cost_reconfig0) * 100,
        "reconfig0_reconfig_average": numpy.mean(reconfig0_recovery_list),

        "reconfig2_total_runtime": time_cost_reconfig2,
        "reconfig2_reconfig_overhead": reconfig2_total_recovery,
        "reconfig2_precentage": (reconfig2_total_recovery / time_cost_reconfig2) * 100,
        "reconfig2_reconfig_average": numpy.mean(reconfig2_recovery_list),
        
    }

def gen_latex_config_table(df_selftf, df_baseline):
    target_baseline_worst_job_id, target_baseline_average_job_id, target_selftf_job_id, target_baseline_best_job_id = get_ids_by_dataframe(
        df_selftf, df_baseline)

    config_worst = common.PSTunerConfiguration(py_dict=json.loads(df_baseline.loc[df_baseline["job_id"] == target_baseline_worst_job_id, "pstuner_config"].values[0]))
    config_average = common.PSTunerConfiguration(py_dict=json.loads(df_baseline.loc[df_baseline["job_id"] == target_baseline_average_job_id, "pstuner_config"].values[0]))
    config_selftf = common.PSTunerConfiguration(py_dict=json.loads(df_selftf.loc[df_selftf["job_id"] == target_selftf_job_id, "pstuner_config"].values[0]))
    config_optimal = common.PSTunerConfiguration(py_dict=json.loads(df_baseline.loc[df_baseline["job_id"] == target_baseline_best_job_id, "pstuner_config"].values[0]))

    sequence = ["ps_num", "worker_num", "intra_op_parallelism_threads", "inter_op_parallelism_threads",
                "do_common_subexpression_elimination", "max_folded_constant_in_bytes",
                "do_function_inlining", "global_jit_level", "infer_shapes", "place_pruned_graph",
                "enable_bfloat16_sendrecv"]
    configs_in_sequence = [config_worst, config_average, config_selftf, config_optimal]

    # Construct the list here
    value_list = []

    def regular_wrapper(value):
        return str(value)

    def bool_wrapper(value):
        return str(bool(value))

    def global_jit_level_wrapper(value):
        if value == 0:
            return "OFF"
        elif value == 1:
            return "ON\_1"
        elif value == 2:
            return "ON\_2"
        else:
            raise Exception()

    for x in sequence:
        wrapper = regular_wrapper
        if x =="global_jit_level":
            wrapper = global_jit_level_wrapper
        elif x == "do_common_subexpression_elimination" or \
             x == "do_function_inlining" or \
             x == "infer_shapes" or \
             x == "place_pruned_graph" or \
             x == "enable_bfloat16_sendrecv":
             wrapper = bool_wrapper
        for config in configs_in_sequence:
            if x == "max_folded_constant_in_bytes" and config.__dict__[x] == 1:
                value_list.append(str(10485760))
            else:
                value_list.append(wrapper(config.__dict__[x]))

    ret = """
    \\begin{tabular}{@{}|@{}p{5.8cm}||c|c|c|c|} 
    \hline  
    Knob (names simplified) & {\sf Worst} & {\sf Average} & {\sf TFOnline} & {\sf Best} \\\\\hline\hline
        \emph{ps} & %s & %s & %s & %s \\\\
        \emph{worker} & %s & %s & %s & %s  \\\\
        \emph{intra\_op\_parallelism\_threads} & %s & %s & %s & %s  \\\\
        \emph{inter\_op\_parallelism\_threads} & %s & %s & %s & %s  \\\\
        \emph{do\_common\_subexpression\_elimination} & %s & %s & %s & %s  \\\\
        \emph{max\_folded\_constant\_in\_bytes} & %s & %s & %s & %s  \\\\
        \emph{do\_function\_inlining} & %s & %s & %s & %s  \\\\
        \emph{global\_jit\_level} & %s & %s & %s & %s  \\\\
        \emph{infer\_shapes} & %s & %s & %s & %s  \\\\
        \emph{place\_pruned\_graph} & %s & %s & %s & %s  \\\\
        \emph{enable\_bfloat16\_sendrecv} & %s & %s & %s & %s  \\\\
    \hline
\end{tabular}
        """ % tuple(value_list)

    return ret

def get_baseline_workload_paths():
    return list(map(lambda workload: os.path.join(base_path,"baseline",workload), workloads))

def gen_hardware_efficiency_statistic_efficiency_table(dict_workload_df_tuple):
    """
    :return:
    """
    values = []
    for workload in workloads:

        df_tuple = dict_workload_df_tuple[workload]
        df_selftf = df_tuple[0]
        df_baseline = df_tuple[1]
        target_baseline_worst_job_id, target_baseline_average_job_id, target_selftf_job_id, target_baseline_best_job_id = get_ids_by_dataframe(
            df_selftf, df_baseline)

        worst_record = df_baseline.loc[df_baseline["job_id"]==target_baseline_worst_job_id].head(1)
        average_record = df_baseline.loc[df_baseline["job_id"]==target_baseline_average_job_id].head(1)
        selftf_record = df_selftf.loc[df_selftf["job_id"]==target_selftf_job_id].head(1)
        best_record = df_baseline.loc[df_baseline["job_id"]==target_baseline_best_job_id].head(1)

        sequence = [worst_record, average_record, selftf_record , best_record]
        for record in sequence:
            number_of_iter = record["N_iterations"].values[0]
            total_runtime = record["time_cost"]
            average_iter_time = str("%.3f" %(float(total_runtime) / float(number_of_iter)))
            values.extend([number_of_iter, average_iter_time])


    ret = """
\\begin{tabular}{c|cc|cc|cc|cc|}
\cline{2-9}& \multicolumn{2}{c|}{\sf Worst} & \multicolumn{2}{|c|}{\sf Average} & \multicolumn{2}{|c|}{\\name} & \multicolumn{2}{|c|}{\sf Best}   \\\\
\cline{2-9}& \# of & time per  &  \# of  & time per &  \# of  & time per &  \# of  & time per \\\\
&  iterations  & iteration  &  iterations  &  iteration &  iterations  & iteration  &  iterations  &  iteration \\\\\hline
\multicolumn{1}{|l|}{LogR}  &      %s  &  %ss    & %s    &  %ss  &      %s  &  %ss    & %s    &  %ss   \\\\
\multicolumn{1}{|l|}{SVM}   &      %s  &  %ss    & %s    &  %ss  &      %s  &  %ss    & %s    &  %ss    \\\\
\multicolumn{1}{|l|}{CNN on CIFAR}  &      %s  &  %ss    & %s    &  %ss  &      %s  &  %ss    & %s    &  %ss     \\\\
\multicolumn{1}{|l|}{CNN on ImageNet}  &      %s  &  %ss    & %s    &  %ss  &      %s  &  %ss    & %s    &  %ss     \\\\\hline
\end{tabular}

    """ % tuple(values)
    return ret


def gen_reconfig_details_table(reconfig_collection):
    # "reconfig0_total_runtime":time_cost_reconfig0,
    # "reconfig0_reconfig_overhead":reconfig0_total_recovery,
    # "reconfig0_precentage":(
    #                                reconfig0_total_recovery / time_cost_reconfig0) * 100,
    # "reconfig0_reconfig_average": numpy.mean(reconfig0_recovery_list),
    #
    # "reconfig2_total_runtime": time_cost_reconfig2,
    # "reconfig2_reconfig_overhead": reconfig2_total_recovery,
    # "reconfig2_precentage": (
    #                                 reconfig2_total_recovery / time_cost_reconfig2) * 100,
    # "reconfig2_reconfig_average": numpy.mean(reconfig2_recovery_list),
    value_sequence = ["num_do_reconfig", "reconfig0_reconfig_overhead","reconfig2_reconfig_overhead",
                      "reconfig0_reconfig_average", "reconfig2_reconfig_average"]

    values = []
    for workload in workloads:
        for value_idx in value_sequence:
            if value_idx == "num_do_reconfig":
                values.append(str(int(reconfig_collection[workload][value_idx])))
            elif "precentage" in value_idx:
                values.append("%.1f" % reconfig_collection[workload][value_idx])
            else:
                values.append("%.0f" % reconfig_collection[workload][value_idx])

    ret = """
\\begin{tabular}{|c|c|c|c||c|c|}\hline
Workload & \# of Reconfig & \multicolumn{2}{c||}{(a) Total Overhead}  & \multicolumn{2}{c|}{(b) Overhead per reconfiguration} \\\\\cline{3-6}
 &  & {\sf Baseline}  & {\sf \\name}  &  {\sf Baseline}  & {\sf \\name} \\\\\hline\hline
LogR  &      %s  &  %ss    & %ss    &  %ss  &      %ss        \\\\
SVM   &      %s  &  %ss    & %ss    &  %ss  &      %ss      \\\\
CNN on CIFAR  &      %s  &  %ss    & %ss    &  %ss  &      %ss       \\\\
CNN on ImageNet  &      %s  &  %ss    & %ss    &  %ss  &      %ss     \\\\\hline
\end{tabular}
""" % tuple(values)
    return ret

def gen_rank_table(baseline_paths):
    """
    average_mre_runtime	 average_rank_runtime	best_rank_runtime	model	worst_rank_runtime
    :param baseline_paths:
    :return:
    """
    # # !!! Estimation function rank
    summary_df = script_error_rank.main(
        baseline_paths,
        [(0, 108), (108, 216), (216, 324)]
    )
    values = []
    value_sequence = ["average_rank_runtime", "best_rank_runtime", "worst_rank_runtime", "average_mre_runtime"]
    for value_idx in value_sequence:
        for workload in workloads:
            value = summary_df.loc[summary_df["model"] == workload][value_idx].values[0]
            if value_idx == "average_rank_runtime" or value_idx == "average_mre_runtime":
                values.append("%.1f" % value)
            else:
                values.append(int(value))
    ret = """
\\begin{tabular}{|c||c|c|c|c|}\hline
Workload & LogR & SVM & CNN on CIFAR & CNN on ImageNet\\\\ \hline\hline
Rank & %s & %s & %s & %s \\\\ \hline
%% Best Rank & %s & %s & %s & %s \\\\ \hline
%% Worst Rank & %s & %s & %s & %s\\\\ \hline
%% \\bf MRE & %s & %s & %s & %s \\\\ \hline
\end{tabular}
    """ % tuple(values)
    return ret

if __name__ == "__main__":


    regex_config_output = re.compile("Dry Run: learning_rate: (.*) ,batch_size: (.*) ,optimizer: (.*) ,number_worker: (.*) ,number_ps: (.*) ,num_intra_thread: (.*) ,num_inter_thread: (.*)")

    monitor_file = "monitor.out"
    summary_csv = "summary.csv"

    reconfig_collection = {}

    baseline_paths = []

    """
    CNN : (df_seltf, df_baseline}
    """
    dict_workload_df_tuple = {}

    for workload in workloads:
        folder_name = workload
        baseline_path = os.path.join(base_path,"baseline", folder_name)
        baseline_paths.append(baseline_path)
        selftf_path = os.path.join(base_path, "selftf", folder_name)
        selftf_reconfig_0_path = os.path.join(base_path, "selftf_reconfig_0", folder_name)

        df_baseline = get_dataframe_baseline(baseline_path)
        df_selftf = get_dataframe_selftf_or_reconfig0(selftf_path)
        df_selftf_reconfig0 = get_dataframe_selftf_or_reconfig0(selftf_reconfig_0_path, skip_config=True)

        dict_workload_df_tuple[workload] = (df_selftf, df_baseline)

        # For graph completion time
        dict_complete_time = get_completiton_time_gnu_data(df_baseline, df_selftf)
        args = dict(dict_complete_time)
        args["path"] = os.path.join(gnu_plot_completition_time_path, dict_completition_time_target_file[workload])
        write_completion_time_gnu_plot(**args)

        # # For loss graph
        get_loss_graph(df_selftf, df_baseline)

        # For table reconfiguration time
        ret = get_reconfig_table(df_selftf, df_selftf_reconfig0)
        reconfig_collection[workload] = ret

        ## gen reconfiguration table
        table_latex = gen_latex_config_table(df_selftf, df_baseline)
        with open(os.path.join(paper_table_base_path, "config_details_"+ workload + ".tex"), "w") as f:
            f.write(table_latex)

    # trigger pdf generation
    subprocess.check_output(
        ["sh", os.path.join(gnu_plot_completition_time_path, "gen_pdf.sh")]
    )
    subprocess.check_output(["find", gnu_plot_completition_time_path, "-name", "*.pdf", "-exec", "cp", "{}",paper_image_base_path,";"])
    logging.info("Finish update completion time graph")

    # Hardware efficiency table generation
    latex_hardware_efficiency = gen_hardware_efficiency_statistic_efficiency_table(dict_workload_df_tuple)
    with open(os.path.join(paper_table_base_path,
                           "hardware_efficiency_table.tex"), "w") as f:
        f.write(latex_hardware_efficiency)
    logging.info("Finish update hardware efficiency graph")

    latex_reconfig_detail_table = gen_reconfig_details_table(reconfig_collection)
    with open(os.path.join(paper_table_base_path,
                           "reconfig_detail_table.tex"), "w") as f:
        f.write(latex_reconfig_detail_table)
    logging.info("Finish update reconfig detail graph")

    latex_rank_table = gen_rank_table(get_baseline_workload_paths())
    with open(os.path.join(paper_table_base_path,
                           "rank_table.tex"), "w") as f:
        f.write(latex_rank_table)
    logging.info("Finish update rank table graph")




