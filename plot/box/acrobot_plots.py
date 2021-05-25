import os
import sys
cwd = os.getcwd()
sys.path.insert(0, cwd+'/../..')
from plot.box.utils_plot import *
from plot.box.paths_acrobot import *

def top_param():
    '''
    calibration = {
        "knn": ac_knn_5k,
        "knn(laplace)": ac_laplace_knn_5k,
        # "network": ac_network_5k,
        # "network(laplace)": ac_laplace_network_5k,
        "network(scaled)": ac_scale_network_5k,
        "network(scaled+laplace)": ac_scale_laplace_network_5k,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    fqi = {"fqi": ac_fqi_eps0p1}
    plot_compare_top(te, calibration, fqi, random, "totals", "../img/acrobot_top_fqi0.1", outer=30, res_scale=-1)

    calibration = {
        "15k knn": ac_knn_15k,
        # "10k knn": ac_knn_10k,
        "5k knn": ac_knn_5k,
        "2.5k knn": ac_knn_2p5k,
        # "1k knn": ac_knn_1k,
        # "500 knn": ac_knn_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_knn_size", outer=30, res_scale=-1, ylim=[80, 300])

    calibration = {
        "15k knn(laplace)": ac_laplace_knn_15k,
        "10k knn(laplace)": ac_laplace_knn_10k,
        "5k knn(laplace)": ac_laplace_knn_5k,
        "2.5k knn(laplace)": ac_laplace_knn_2p5k,
        "1k knn(laplace)": ac_laplace_knn_1k,
        "500 knn(laplace)": ac_laplace_knn_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_knn_laplace_size", outer=30, res_scale=-1, ylim=[80, 300])

    calibration = {
        "15k network(laplace)": ac_scale_laplace_network_15k,
        "10k network(laplace)": ac_scale_laplace_network_10k,
        "5k network(laplace)": ac_scale_laplace_network_5k,
        "2.5k network(laplace)": ac_scale_laplace_network_2p5k,
        "1k network(laplace)": ac_scale_laplace_network_1k,
        "500 network(laplace)": ac_scale_laplace_network_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_network_laplace_size", outer=30, res_scale=-1, ylim=[80, 1000])

    calibration = {
        "15k network": ac_scale_network_15k,
        # "10k network": ac_scale_network_10k,
        "5k network": ac_scale_network_5k,
        "2.5k network": ac_scale_network_2p5k,
        # "1k network": ac_scale_network_1k,
        # "500 network": ac_scale_network_500,
    }
    random = ac_rnd
    te = {"true": ac_true_temp}
    # plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_network_size", outer=30, res_scale=-1, ylim=[80, 1000])
    '''
    
    

def sweep_model():
    calibration = {
        "bad (network)": ac_subsuboptim_network,
        "average (network)": ac_suboptim_network,
        "optimal (network)": ac_optim_network,
        "bad (knn)": ac_subsuboptim_knn,
        "average (knn)": ac_suboptim_knn,
        "optimal (knn)": ac_optim_knn
    }
    te = {"true": ac_true}
    #fqi = {"fqi": ac_fqi}
    #cem = {"cem": ac_cem}
    plot_generation(te, calibration, ranges, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000, res_scale=-1)
    #plot_each_run(te, calibration, "totals", "../img/acrobot_model", outer=30, sparse_reward=-1, max_len=1000)

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

def dqn():
    calibration = {
        "5k knn(laplace)": ac_laplace_knn_5k_dqn,
    }
    random = np.array(ac_rnd) % 24
    te = {"true": ac_true_dqn}
    plot_compare_top(te, calibration, None, random, "totals", "../img/acrobot_dqn", outer=30, res_scale=-1)

if __name__ == '__main__':
    ranges = [0]
    top_param()
    # dqn()
    # sweep_model()
    # data_density()
    #top_param()
