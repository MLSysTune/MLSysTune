# Result Layout
# JobID, Estimation function segment, actual remaining iteration, actual remaining time, estimated remaining iteration, estimated remaining time, time error, actual rank, estimated rank
import argparse
import csv
import logging
import os
import time

import pandas
import re
import sys

import selftf
from selftf.lib.common import PSTunerTrainingData
from selftf.lib.tuner import TensorFlowConfigurationManager, TFConfigUtil, \
    GPTrainingModel

matrix_timestamp_idx = 0
matrix_loss_idx = 1
matrix_iteration_idx = 2
matrix_elapsed_time_sec = 3

gnu_plot_base_path = "/Users/cyliu/Documents/selftf_paper/gnuplot_script"
folder_filter_regex = re.compile("(.*)_(.*)_(.*)")

class ResultObj:

    def __init__(self, job_id, estimated_segment, actual_remaining_iteration,
        actual_remaining_time, estimated_remaining_iteration,
        estimated_remaining_time, ml_job_name,
        actual_average_iteration_time,
        estimated_average_iteration_time):
        self.job_id = job_id
        self.estimated_segment = estimated_segment
        self.actual_remaining_iteration = actual_remaining_iteration
        self.actual_remaining_time = actual_remaining_time
        self.actual_average_iteration_time=actual_average_iteration_time
        self.estimated_remaining_iteration = estimated_remaining_iteration
        self.estimated_remaining_time = estimated_remaining_time
        self.estimated_average_iteration_time=estimated_average_iteration_time
        self.ml_job_name = ml_job_name
        self.iteration_error = .0
        self.time_error = .0
        self.actual_rank = .0
        self.estimated_rank = .0


def proces_ml_job_log(folder_path, list_estimated_range):
    """
    :param str folder_path:
    :return:
    """
    ret = []
    try:

        job_id = folder_path[folder_path.rindex("/")+1:]
        path_training_data_csv = os.path.join(folder_path, job_id+'iteration_training_data.csv')
        summary_csv_path = os.path.join(folder_path, "summary.csv")
        summary_obj = get_summary_object(summary_csv_path)


        if summary_obj.runtime_sec == 0:
            return ret
        with open(path_training_data_csv, "r") as f:
            list_training_data = get_list_training_data(f)
            for estimated_range in list_estimated_range:
                ret.append(
                    process_segment(list_training_data, estimated_range,
                    epsilon=float(summary_obj.target_loss),
                    job_id=job_id))
    except:
        logging.exception("can't process %s"%folder_path)
    return ret

def get_summary_object(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        summary = selftf.lib.common.Summary()
        summary.__dict__ = reader.next()
        return summary


def get_list_training_data(f):
    dict_reader = csv.DictReader(f)
    ret = []
    for obj in dict_reader:
        ret.append((float(obj["timestamp"]),
                    float(obj["loss"]),
                    int(obj["step"]),
                    float(obj["elapsed_time_sec"])))
    return ret

def process_segment(training_data_matrix, target_segment, epsilon, job_id):
    ml_job_name = folder_filter_regex.search(job_id).group(1)
    final_timestamp = training_data_matrix[-1][matrix_timestamp_idx]
    final_iteration_idx = training_data_matrix[-1][matrix_iteration_idx]

    target_training_data = training_data_matrix[target_segment[0]:target_segment[1]]

    tfcm = TensorFlowConfigurationManager(lambda: 4, lambda: 16, (0.00001, 0.0001), (1000, 10000))
    tcu = TFConfigUtil(tfcm)

    gp = GPTrainingModel(tcu, epsilon=epsilon)
    estimation_result = gp.estimate_remaining_time(matrix_to_list_training_data(target_training_data))

    actual_remaining_time = final_timestamp - training_data_matrix[target_segment[1]-1][matrix_timestamp_idx]
    actual_average_iteration_time = actual_remaining_time / len(training_data_matrix)
    ret = ResultObj(job_id, target_segment,
              actual_remaining_iteration=final_iteration_idx - target_segment[1]+1,
              actual_remaining_time=final_timestamp - training_data_matrix[target_segment[1]-1][matrix_timestamp_idx],
              actual_average_iteration_time=actual_average_iteration_time,
              estimated_remaining_iteration= estimation_result.remaining_iteration,
              estimated_remaining_time=estimation_result.remaining_time,
              estimated_average_iteration_time=estimation_result.average_iteration_time,
              ml_job_name=ml_job_name
              )
    return ret

def matrix_to_list_training_data(matrix):
    ret = []
    for row in matrix:
        training_data = PSTunerTrainingData(
            loss=row[matrix_loss_idx],
            step=row[matrix_iteration_idx],
            timestamp=row[matrix_timestamp_idx],
            elapsed_time_in_ms=row[matrix_elapsed_time_sec]
        )
        ret.append(training_data)

    return ret

def calculate_all_training_error_and_rank(list_result_obj):
    dataframe = pandas.DataFrame(list(map(lambda a:a.__dict__, list_result_obj)),
                                 columns=["job_id",
                                          "estimated_segment",
                                          "actual_remaining_iteration",
                                          "actual_remaining_time",
                                          "actual_average_iteration_time",
                                          "estimated_remaining_iteration",
                                          "estimated_remaining_time",
                                          "estimated_average_iteration_time",
                                          "ml_job_name"])
    ml_job_segment_dataframe = dataframe.groupby(["ml_job_name", "estimated_segment"])
    inter_ret = []
    temp_base_path = os.path.join("/tmp/", "analysis_"+str(time.time())) + dataframe["ml_job_name"].min()
    os.mkdir(temp_base_path)
    for ml_job_segment, gp in ml_job_segment_dataframe:
        dataframe = calculdate_error_and_rank(gp)

        temp_folder_path = os.path.join(temp_base_path, "%s_%s-%s" % (ml_job_segment[0],
                                                                      str(ml_job_segment[1][0]),
                                                                      str(ml_job_segment[1][1])))
        os.mkdir(temp_folder_path)
        with open(os.path.join(temp_folder_path, "raw_table.csv"), "w") as f:
            dataframe.to_csv(f)

        mre_runtime = dataframe["remaining_runtime_error"].mean()
        mre_iteration = dataframe["remaining_iteration_error"].mean()
        mre_iteration_time = dataframe["average_iteration_time_error"].mean()
        rank_runtime = dataframe[dataframe["estimated_remaining_time_rank"] == 1]["actual_remaining_time_rank"].min()
        rank_iteration = dataframe[dataframe["estimated_remaining_iteration_rank"] == 1]["actual_remaining_iteration_rank"].min()
        inter_ret.append({"ml_job_segment":ml_job_segment,
                    "mre_iteration":mre_iteration,
                    "mre_runtime":mre_runtime,
                    "mre_iteration_time":mre_iteration_time,
                    "rank_runtime":rank_runtime,
                    "rank_iteration":rank_iteration})

    df = pandas.DataFrame(inter_ret)
    ret = {
        "model":dataframe["ml_job_name"].min(),
        "average_mre_runtime": df["mre_runtime"].mean(),
        "best_rank_runtime": df["rank_runtime"].min(),
        "average_rank_runtime": df["rank_runtime"].mean(),
        "worst_rank_runtime": df["rank_runtime"].max(),
    }


    # with open(os.path.join(temp_base_path, "summary.csv"), "w") as f:
    #     writer = csv.DictWriter(f, ["ml_job_segment", "mre_iteration",
    #                                 "mre_runtime","mre_iteration_time","rank_runtime",
    #                                 "rank_iteration"])
    #     writer.writeheader()
    #     writer.writerows(inter_ret)
    # print "output_base_path: %s" % (temp_base_path)
    return inter_ret, ret

def calculdate_error_and_rank(dataframe):
    """
    :param pandas.DataFrame dataframe:
    :return:
    """
    # sort by actual iteration
    dataframe['actual_remaining_iteration_rank'] = \
        dataframe['actual_remaining_iteration'].rank()
    # actual iteration rank
    dataframe['actual_remaining_time_rank'] = \
        dataframe['actual_remaining_time'].rank()
    dataframe['estimated_remaining_iteration_rank'] = \
        dataframe['estimated_remaining_iteration'].rank()
    dataframe['estimated_remaining_time_rank'] = \
        dataframe['estimated_remaining_time'].rank()

    def error_func(actual, estimation):
        return abs((actual - estimation)/actual)

    def remaining_iteration_error(row):
        a = row["actual_remaining_iteration"]
        estimated = row["estimated_remaining_iteration"]
        return error_func(a, estimated)

    def remaining_runtime_error(row):
        a = row["actual_remaining_time"]
        estimated = row["estimated_remaining_time"]
        return error_func(a, estimated)

    def avg_iteration_error(row):
        a = row["actual_average_iteration_time"]
        estimated = row["estimated_average_iteration_time"]
        return error_func(a, estimated)

    dataframe['remaining_iteration_error'] = dataframe.apply(
        lambda row: remaining_iteration_error(row), axis=1
    )
    dataframe['remaining_runtime_error'] = dataframe.apply(
        lambda row: remaining_runtime_error(row), axis=1
    )
    dataframe['average_iteration_time_error'] = dataframe.apply(
        lambda row: avg_iteration_error(row), axis=1
    )
    return dataframe
    # sort by actual remaining runtime

def main(list_paths, list_estimation_range):
    summary_ret = []
    for base_path in list_paths:
        # list all folder
        ret = []
        folders = [f for f in os.listdir(base_path) if os.path.isdir(
            os.path.join(base_path, f)) and folder_filter_regex.search(
            f) is not None]
        for folder in folders:
            ret.extend(proces_ml_job_log(os.path.join(base_path, folder),
                                         list_estimation_range))

        inter_ret, ret = calculate_all_training_error_and_rank(ret)
        summary_ret.append(ret)
    summary_df = pandas.DataFrame(summary_ret)
    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", dest="base_path")
    parser.add_argument("--list_estimation_range", dest="list_estimation_range",default="[(0,100)]")
    args = parser.parse_args()

    base_path_list = eval(args.base_path)
    if not isinstance(base_path_list, list):
        base_path = [base_path_list]

    summary_df = main(base_path_list, eval(args.list_estimation_range))
    logging.info(summary_df.to_string())
    with open(os.path.join(gnu_plot_base_path, "estimation_function_ranks.csv"),
              "w") as f:
        summary_df.to_csv(f)
    sys.exit(0)




