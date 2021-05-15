import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot import *

def top_param():
    calibration = {
        "knn": ac_knn,
        "knn(laplace)": ac_laplace_knn,
        "network": ac_network,
        "network(laplace)": ac_laplace_network,
        "network(scaled)": ac_scale_network,
        "network(scaled+laplace)": ac_scale_laplace_network,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_top", outer=30, res_scale=-1)

    calibration = {
        "15k knn(laplace)": ac_laplace_knn,
        "15k network(scaled+laplace)": ac_scale_laplace_network,

        "10k knn(laplace)": ac_laplace_knn_10k,
        "10k network(scaled+laplace)": ac_scale_laplace_network_10k,

        "5k knn(laplace)": ac_laplace_knn_5k,
        "5k network(scaled+laplace)": ac_scale_laplace_network_5k,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_size", outer=30, res_scale=-1)

    calibration = {
        "15k knn": ac_knn,
        "10k knn": ac_knn_10k,
        "5k knn": ac_knn_5k,
        "1k knn": ac_knn_1k,
        "500 knn": ac_knn_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_knn_size", outer=30, res_scale=-1, ylim=[80, 300])
    calibration = {
        "15k knn(laplace)": ac_laplace_knn,
        "10k knn(laplace)": ac_laplace_knn_10k,
        "5k knn(laplace)": ac_laplace_knn_5k,
        "1k knn(laplace)": ac_laplace_knn_1k,
        "500 knn(laplace)": ac_laplace_knn_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_knn_laplace_size", outer=30, res_scale=-1, ylim=[80, 300])

def sweep_model():
    calibration = {
        "calibration model": ac_offline,
    }
    te = {"true": ac_true}
    #plot_generation(te, calibration, ranges, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    plot_each_run(te, calibration, "totals", "../img/acrobot_run", outer=30, sparse_reward=-1, max_len=1000)

def data_density():
    datasets = {
        "15k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step15k/optimalfixed_eps0/param_0/",
        "10k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step10k/optimalfixed_eps0/param_0/",
        "5k": "../../data/hyperparam_v5/acrobot/offline_data/true_restarts/esarsa/step5k/optimalfixed_eps0/param_0/",
    }
    dimension = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
    }
    group = {"01": [0, 1], "23": [2, 3], "45": [4, 5]}
    key="new state"
    for i in range(10):
        run = "traces-{}".format(i)
        plot_dataset(datasets, key, dimension, group, run, "../img/data_density")

if __name__ == '__main__':
    ranges = [0]
    top_param()
    #sweep_model()
    # data_density()