import json

import pandas as pd

from selftf.lib.common import PSTunerTrainingData, PSTunerConfiguration, \
    PSTunerTrainingDataSerializer

PATH_SOURCE = "cnn_baseline_history.csv"
PATH_DST = "cnn_history.csv"
NUM_MACHINE = 36
NUM_THREAD = 16

workloads = ["cnn", "svm", "lr"]

def generate(workload):
    PATH_SOURCE = "{}_baseline_history.csv".format(workload)
    PATH_DST = "{}_history.csv".format(workload)
    df = pd.read_csv(PATH_SOURCE, sep='\t')
    with open(PATH_DST, 'w') as f:
        for idx, row in df.iterrows():
            config = PSTunerConfiguration(
                num_ps=NUM_MACHINE-row["workers"],
                num_worker=row["workers"],
                intra_op_parallelism_threads=row['n_intra'],
                inter_op_parallelism_threads=NUM_THREAD-row['n_intra'],
                batch_size=100,
                learning_rate=0.0001,
                optimizer="Adam"
            )
            training_data = PSTunerTrainingData(
                ps_config=config,
                elapsed_time_in_ms=row['runtime']
            )
            f.write(json.dumps(training_data, cls=PSTunerTrainingDataSerializer))
            f.write('\n')

for workload in workloads:
    generate(workload)