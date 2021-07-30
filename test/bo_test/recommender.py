import sys

from selftf.lib.gpr.gp_tf import GPRGD
from sklearn.preprocessing import StandardScaler
import numpy as np

from .bo_test import generate_training_data


def recommend_next_x(X_history, loss_history, y_history, num_samples=30):
    MAX_ITER = 500
    MAX_TRAIN_SIZE = 7000
    BATCH_SIZE = 3000
    NUM_THREADS = 4
    DEFAULT_LENGTH_SCALE = 1.0
    DEFAULT_MAGNITUDE = 1.0
    DEFAULT_RIDGE = 1.0
    DEFAULT_LEARNING_RATE = 0.01
    DEFAULT_EPSILON = 1e-6
    DEFAULT_MAX_ITER = 5
    DEFAULT_RIDGE = 1.0
    DEFAULT_SIGMA_MULTIPLIER = 3.0
    DEFAULT_MU_MULTIPLIER = 1.0

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_history)
    #print(X_scaled)
    
    l_scaler = StandardScaler()
    l_scaled = l_scaler.fit_transform(loss_history.reshape(-1,1))
    #print(l_scaled)
    
    X_l_scaled = np.column_stack((X_scaled, l_scaled))
    #print(X_l_scaled)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_history.reshape(-1,1))

    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        X_min[i] = col_min
        X_max[i] = col_max
        X_samples[:, i] = np.random.rand(num_samples) * (col_max - col_min) + col_min


    model = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
                      magnitude=DEFAULT_MAGNITUDE,
                      max_train_size=MAX_TRAIN_SIZE,
                      batch_size=BATCH_SIZE,
                      num_threads=NUM_THREADS,
                      learning_rate=DEFAULT_LEARNING_RATE,
                      epsilon=DEFAULT_EPSILON,
                      max_iter=MAX_ITER,
                      sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER,
                      mu_multiplier=DEFAULT_MU_MULTIPLIER)
    model.fit(X_l_scaled, y_scaled, X_min, X_max, ridge=DEFAULT_RIDGE)
    #print(l_scaler.transform(loss_history[-1]))
    current_loss = l_scaler.transform(loss_history[-1])[0][0]
    res = model.predict(X_samples, current_loss)
    

    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]

    print("Raw gp ret:%s " % str(best_config))

    best_config = X_scaler.inverse_transform(best_config)

    # Although we have max/min limits in the GPRGD training session, it may
    # lose some precisions. e.g. 0.99..99 >= 1.0 may be True on the scaled data,
    # when we inversely transform the scaled data, the different becomes much larger
    # and cannot be ignored. Here we check the range on the original data
    # directly, and make sure the recommended config lies within the range
    X_min_inv = X_scaler.inverse_transform(X_min)
    X_max_inv = X_scaler.inverse_transform(X_max)
    best_config = np.minimum(best_config, X_max_inv)
    best_config = np.maximum(best_config, X_min_inv)
    
    best_ei = -res.minl[best_config_idx]
    best_ei = l_scaler.scale_ * best_ei
    return best_config, best_ei

if __name__ == "__main__":
    unnormalize_x, unnormalize_y = generate_training_data(
        "cnn_testing_result.csv")

    unnormalized_x_without_loss = np.delete(unnormalize_x,
                                            unnormalize_x.shape[1] - 1, 1)
    unnormalize_l = unnormalize_x[:, -1]

    ret = recommend_next_x(unnormalized_x_without_loss, unnormalize_l, unnormalize_y.reshape(1,-1))
    print(ret)
    sys.exit(0)