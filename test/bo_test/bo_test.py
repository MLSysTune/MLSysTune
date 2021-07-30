# import training data
import csv
import logging

import numpy as np

from selftf.lib.common import Job, \
    get_default_learning_rate_batch_size_optimizer, BaselineRecord
from selftf.lib.common_conf import optimizer_list
from selftf.lib.tuner import TensorFlowConfigurationManager, TFConfigUtil, \
    GPTrainingModel

_num_thread=16
_num_machine=36
def generate_training_data(file_name):
    ret = []
    list_x = []
    list_y = []
    with open(file_name, "r") as file:
        reader = csv.DictReader(file)
        for record_json in reader:
            record = BaselineRecord(**record_json)
            num_ps=_num_machine-int(record.N_workers)
            intra_threa = int(record.N_intra)
            loss = float(record.first_loss)
            runtime = float(record.time_cost)

            list_x.append([num_ps, intra_threa, loss])
            list_y.append([runtime])

    return np.array(list_x), np.array(list_y)

if __name__ == "__main__":
    file_name = "svm_testing_data_for_bo_with_progress.csv"
    unnormalize_x, unnormalize_y = generate_training_data(file_name)

    tfcm = TensorFlowConfigurationManager(lambda: _num_machine, lambda: _num_thread, (0.00001, 0.0001), (1000, 10000))
    tcu = TFConfigUtil(tfcm)

    gp = GPTrainingModel(tcu)
    unnormalized_x_without_loss = np.delete(unnormalize_x, unnormalize_x.shape[1]-1, 1)
    unnormalize_l = np.reshape(unnormalize_x[:,-1], (-1,1))
    normalized_x_without_loss = tcu.x_scalar.fit_transform(unnormalized_x_without_loss)
    normalized_l = tcu.l_scaler.fit_transform(unnormalize_l)
    normalize_y = tcu.y_scalar.fit_transform(unnormalize_y)
    normalized_x_with_loss = np.column_stack((normalized_x_without_loss, normalized_l))
    gp.train_gpr(normalized_x_with_loss, normalize_y)

    model_name = file_name[:file_name.index("_")].upper()
    learnining_rate, batch_size, optimizer = get_default_learning_rate_batch_size_optimizer(model_name)
    mock_job_obj = Job(
        learning_rate=learnining_rate,
        batch_size=batch_size,
        optimizer=optimizer_list.index(optimizer)
    )
    config_obj = gp.get_best_config(job_obj=mock_job_obj,last_loss=unnormalize_l[-1][0])
    logging.info(config_obj)